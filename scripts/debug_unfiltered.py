import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from config.settings import settings

store = DocumentStore(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
r = Retriever(store)
r.min_relevance = 0.0   # switch both gates off to see the raw ranking
r.min_coverage = 0.0

for q in [
    "What limitations are reported for Self-Refine?",
    "What GPU was used for training?",
]:
    print("\nQUESTION:", q)
    print("GROUPS:", r._query_groups(q))
    for item in r.search(q, top_k=5):
        text = item["text"][:150].replace("\n", " ")
        print(item["chunk_id"], "| cov", item["coverage"], "| sim", item["similarity"])
        print("    ", text)
