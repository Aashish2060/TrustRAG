import pytest

from src.ingestion.document_store import DocumentStore


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Every test gets a throwaway document store, so tests never write
    into data/runtime/documents.json."""
    from src.api import main

    monkeypatch.setattr(
        main, "store",
        DocumentStore(str(tmp_path), chunk_size=1200, chunk_overlap=100),
    )
