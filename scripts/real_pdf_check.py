import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient, GenerationError
from src.verification.correction_controller import CorrectionController

p = Path(os.environ.get("PDF_PATH", ""))
if not p.is_file():
    raise SystemExit(f"PDF not found: {p}  (set $env:PDF_PATH first)")
QUESTIONS = [q.strip() for q in os.environ.get("QS", "").split("||") if q.strip()]
if not QUESTIONS:
    raise SystemExit("No questions. Set $env:QS first.")

llm = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model, timeout=60.0)
if not llm.configured:
    raise SystemExit("LLM not configured. Check .env.")

with tempfile.TemporaryDirectory() as d:
    store = DocumentStore(d, settings.chunk_size, settings.chunk_overlap)
    info = store.add(p.name, p.read_bytes())
    print("INGESTED:", info)
    if not store.chunks:
        raise SystemExit("No text extracted. Is the PDF scanned images?")
    titles = sorted({c.get("section_title", "") for c in store.chunks if c.get("section_title")})
    print("section titles (first 8):", titles[:8])
    lens = [len(c["text"]) for c in store.chunks]
    print("chunk length min/avg/max:", min(lens), sum(lens) // len(lens), max(lens))

    controller = CorrectionController(Retriever(store, settings.min_relevance), llm, settings.max_corrections)
    totals = {"supported": 0, "insufficient_evidence": 0, "abstained": 0, "errors": 0}

    for q in QUESTIONS:
        print("\n" + "=" * 72)
        print("Q:", q)
        try:
            r = controller.run(q, top_k=settings.top_k, mode="self_correcting")
        except GenerationError as e:
            totals["errors"] += 1
            print("GENERATION ERROR:", e)
            continue

        print("retrieved:", [(e["chunk_id"][:6], e.get("section_title", "")[:25], e.get("similarity")) for e in r["evidence"]])
        print("STATUS:", r["answer_status"], "| ABSTAINED:", r["abstained"], "| RELIABILITY:", r["reliability_score"])
        print("ATTEMPTS:", json.dumps(r["correction_attempts"]))
        print("ANSWER:", r["final_answer"][:700])
        if r["abstained"]:
            totals["abstained"] += 1
        for c in r["claims"]:
            totals[c["label"]] = totals.get(c["label"], 0) + 1
            flag = "" if c["label"] == "supported" else "   <-- CHECK"
            print(f"- [{c['label']}] score={c['score']} missing={c['missing_numbers']} warn={c['negation_warning']}{flag}")
            print("    ", c["claim"][:300])

    print("\nTOTALS:", totals)
