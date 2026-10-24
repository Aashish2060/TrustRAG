from src.evaluation.metrics import summarize
from src.evaluation.dataset import load_dataset

def test_summary_denominators():
    out = summarize([{"abstained": False, "latency_seconds": 0.2, "claims": [{"label":"supported"}]}])
    assert out["answer_coverage"] == 1
    assert out["claim_support_rate"] == 1

def test_starter_dataset_schema():
    rows = load_dataset()
    assert rows and all(r["synthetic"] for r in rows)
