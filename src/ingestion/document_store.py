from pathlib import Path
import json
import os
import uuid
from threading import RLock
from src.ingestion.loaders import extract_text
from src.ingestion.chunking import chunk_records


class DocumentStore:
    def __init__(self, data_dir="data", chunk_size=1200, chunk_overlap=100):
        self.root = Path(data_dir) / "runtime"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "documents.json"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lock = RLock()
        self.documents = {}
        self.chunks = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                obj = json.loads(self.path.read_text(encoding="utf-8"))
                self.documents = obj.get("documents", {})
                self.chunks = obj.get("chunks", [])
            except (OSError, json.JSONDecodeError):
                # Keep the unreadable file for inspection instead of
                # letting the next save silently overwrite it.
                try:
                    self.path.replace(self.path.with_suffix(".json.corrupt"))
                except OSError:
                    pass
                self.documents, self.chunks = {}, []

    def _save(self):
        # Write to a temp file, then swap it in atomically, so a crash
        # mid-write cannot leave a half-written documents.json.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"documents": self.documents, "chunks": self.chunks},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def add(self, filename: str, content: bytes):
        records = extract_text(filename, content)
        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_records(
            doc_id,
            Path(filename).name,
            records,
            self.chunk_size,
            self.chunk_overlap,
        )

        with self.lock:
            self.documents[doc_id] = {
                "document_id": doc_id,
                "filename": Path(filename).name,
                "chunk_count": len(chunks),
                "characters": sum(len(r["text"]) for r in records),
            }
            self.chunks.extend(chunks)
            self._save()

        return self.documents[doc_id]

    def list(self):
        return list(self.documents.values())

    def delete(self, document_id: str):
        with self.lock:
            if document_id not in self.documents:
                return False

            del self.documents[document_id]
            self.chunks = [
                c for c in self.chunks
                if c["document_id"] != document_id
            ]
            self._save()
            return True

    def reset(self):
        with self.lock:
            self.documents, self.chunks = {}, []
            self._save()