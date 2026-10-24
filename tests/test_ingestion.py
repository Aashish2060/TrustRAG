import pytest
from src.ingestion.loaders import extract_text
from src.ingestion.chunking import chunk_records

def test_txt_ingestion():
    assert "hello" in extract_text("a.txt", b"hello world")[0]["text"]

def test_empty_and_unsupported_files():
    with pytest.raises(ValueError): extract_text("a.txt", b"")
    with pytest.raises(ValueError): extract_text("a.exe", b"hello")

def test_chunking_metadata_and_overlap():
    text = "A" * 160
    chunks = chunk_records("doc1", "a.txt", [{"page": 1, "text": text}], 80, 20)
    assert len(chunks) >= 2
    assert chunks[0]["document_id"] == "doc1"
    assert chunks[0]["filename"] == "a.txt"
