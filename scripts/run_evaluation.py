import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from config.settings import settings
from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.verification.correction_controller import CorrectionController
from src.evaluation.dataset import load_dataset
from src.evaluation.metrics import summarize

def main():
    data_dir = settings.data_dir
    store = DocumentStore(data_dir, settings.chunk_size, settings.chunk_overlap)
    sample = ROOT / "data/sample_documents/renewable_energy.txt"
    # Use a fresh in-memory store for deterministic evaluation, independent of user uploads.
    store.reset()
    store.add(sample.name, sample.read_bytes())
    dataset = load_dataset(str(ROOT / "data/evaluation/starter_dataset.json"))
    client = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    all_rows, summary_rows = [], []
    for mode in ["baseline", "verified", "self_correcting"]:
        run_rows = []
        for item in dataset:
            try:
                result = CorrectionController(Retriever(store, settings.min_relevance), client, settings.max_corrections).run(item["question"], settings.top_k, mode)
                result.update({"example_id": item["example_id"], "category": item["category"], "expected_answerable": item["answerable"], "synthetic_dataset": item["synthetic"]})
                result["abstained"] = "cannot answer" in result["final_answer"].lower() or "insufficient evidence" in result["final_answer"].lower()
                run_rows.append(result)
            except Exception as exc:
                run_rows.append({"example_id": item["example_id"], "mode": mode, "error": exc.__class__.__name__, "latency_seconds": 0, "claims": [], "abstained": True})
        metrics = summarize(run_rows)
        summary_rows.append({"configuration": mode, **metrics})
        all_rows.extend(run_rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out / "metric_comparison.csv", index=False)
    (out / "evaluation_results.json").write_text(json.dumps({"dataset": "synthetic starter dataset", "note": "Prototype smoke-test metrics; lexical verifier and tiny synthetic dataset do not support general scientific conclusions.", "summary": summary_rows, "runs": all_rows}, indent=2), encoding="utf-8")
    frame = pd.DataFrame(summary_rows)
    ax = frame.plot(x="configuration", y=["mean_latency_seconds", "claim_support_rate", "answer_coverage"], kind="bar", rot=0)
    ax.set_title("TrustRAG starter evaluation (synthetic smoke test)")
    ax.set_ylabel("Metric value (different scales; interpret cautiously)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out / "metric_comparison.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nWrote {out / 'metric_comparison.csv'}, evaluation_results.json, metric_comparison.png")

if __name__ == "__main__":
    main()
