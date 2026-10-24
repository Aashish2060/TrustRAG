import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from config.settings import settings
from src.core.models import QueryRequest
from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient, GenerationError
from src.verification.correction_controller import CorrectionController

logger = logging.getLogger("trustrag")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="TrustRAG API", version="0.1.0", description="Research prototype for evidence-grounded RAG verification and bounded self-correction.")
store = DocumentStore(settings.data_dir, settings.chunk_size, settings.chunk_overlap)


def pipeline():
    return CorrectionController(Retriever(store, settings.min_relevance), LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model), settings.max_corrections)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(store.documents), "chunks": len(store.chunks), "generation_mode": "api" if settings.llm_base_url and settings.llm_api_key and settings.llm_model else "local_extractive"}


@app.get("/api/documents")
def list_documents():
    return {"documents": store.list(), "count": len(store.documents)}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # Read at most one byte past the limit, then reject oversized files.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
    try:
        return store.add(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail="Document processing failed") from exc


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    if not store.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "document_id": document_id}


@app.post("/api/index/rebuild")
def rebuild_index():
    # TF-IDF vectors are constructed per query; no separate persistent vector index is needed.
    return {"status": "ready", "documents": len(store.documents), "chunks": len(store.chunks)}


@app.post("/api/query")
def query(request: QueryRequest):
    try:
        return pipeline().run(request.question, request.top_k, request.mode)
    except GenerationError as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from exc


@app.post("/api/query/baseline")
def query_baseline(request: QueryRequest):
    request.mode = "baseline"
    return query(request)


@app.post("/api/query/verified")
def query_verified(request: QueryRequest):
    request.mode = "verified"
    return query(request)


@app.post("/api/query/self-correcting")
def query_self_correcting(request: QueryRequest):
    request.mode = "self_correcting"
    return query(request)


@app.post("/api/evaluation/run")
def run_evaluation():
    return JSONResponse({"status": "not_started", "message": "Use `python scripts/run_evaluation.py` to run the local reproducible starter evaluation."}, status_code=202)