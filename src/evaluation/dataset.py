import json
from pathlib import Path

def load_dataset(path="data/evaluation/starter_dataset.json"):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"example_id", "question", "answerable", "category", "synthetic"}
    for row in records:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Dataset record missing fields: {sorted(missing)}")
    return records
