from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.verification.correction_controller import CorrectionController

def test_pipeline_runs_and_keeps_audit_trail(tmp_path):
    store = DocumentStore(str(tmp_path), chunk_size=100, chunk_overlap=10)
    store.add("facts.txt", b"A kiwi is a flightless bird native to New Zealand. Kiwis are nocturnal.")
    result = CorrectionController(Retriever(store, min_relevance=0), LLMClient(), max_corrections=1).run("What is a kiwi?", top_k=2, mode="self_correcting")
    assert result["initial_answer"]
    assert result["final_answer"]
    assert "correction_attempts" in result
    assert result["mode"] == "self_correcting"
