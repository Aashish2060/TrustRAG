import json
import sys
import tempfile
import traceback
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
    sample = ROOT / "data/sample_documents/renewable_energy.txt"
    dataset = load_dataset(str(ROOT / "data/evaluation/starter_dataset.json"))
    client = LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    all_rows, summary_rows = [], []

    # Evaluation runs on a throwaway store in a temp folder, so it never
    # touches (or deletes) the documents you uploaded to data/runtime.
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = DocumentStore(tmp_dir, settings.chunk_size, settings.chunk_overlap)
        store.add(sample.name, sample.read_bytes())

        for mode in ["baseline", "verified", "self_correcting"]:
            run_rows = []
            for item in dataset:
                try:
                    result = CorrectionController(
                        Retriever(store, settings.min_relevance),
                        client,
                        settings.max_corrections,
                    ).run(item["question"], settings.top_k, mode)
                    result.update({
                        "example_id": item["example_id"],
                        "category": item["category"],
                        "expected_answerable": item["answerable"],
                        "synthetic_dataset": item["synthetic"],
                    })
                    answer = result["final_answer"].lower()
                    result["abstained"] = bool(result.get("abstained")) or (
                        "cannot answer" in answer or "insufficient evidence" in answer
                    )
                    run_rows.append(result)
                except Exception as exc:
                    print(f"[{mode}] {item['example_id']} failed:", file=sys.stderr)
                    traceback.print_exc()
                    run_rows.append({
                        "example_id": item["example_id"],
                        "mode": mode,
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "latency_seconds": 0,
                        "claims": [],
                        "abstained": True,
                    })
            metrics = summarize(run_rows)
            summary_rows.append({"configuration": mode, **metrics})
            all_rows.extend(run_rows)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out / "metric_comparison.csv", index=False)
    (out / "evaluation_results.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic starter dataset",
                "note": "Prototype smoke-test metrics; lexical verifier and tiny synthetic dataset do not support general scientific conclusions.",
                "summary": summary_rows,
                "runs": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(summary_rows)
    ax = frame.plot(
        x="configuration",
        y=["mean_latency_seconds", "claim_support_rate", "answer_coverage"],
        kind="bar",
        rot=0,
    )
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