from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_document_upload_and_list():
    response = client.post("/api/documents/upload", files={"file": ("test.txt", b"The moon orbits Earth.", "text/plain")})
    assert response.status_code == 200
    listing = client.get("/api/documents")
    assert listing.status_code == 200
    assert any(d["filename"] == "test.txt" for d in listing.json()["documents"])
