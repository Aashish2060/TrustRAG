import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from config.settings import settings

store = DocumentStore(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
print("Loaded documents:", len(store.documents), "| chunks:", len(store.chunks))

retriever = Retriever(store, min_relevance=settings.min_relevance)

questions = [
    "What does Self-Refine report about average improvement?",
    "How does intrinsic self-correction affect reasoning tasks?",
    "What limitations are reported for Self-Refine?",
    "Explain Self-Refine and intrinsic self-correction, including the reported improvement and the limitations found in reasoning tasks.",
    "What GPU was used for training?",
]

for q in questions:
    print("\nQUESTION:", q)
    results = retriever.search_compound(q, top_k=10)
    if not results:
        print("  (no chunks passed the filters)")
    for r in results:
        print("  ", r["chunk_id"], "| cov", r["coverage"], "| sim", r["similarity"], "| sub:", r.get("matched_subquery"))
