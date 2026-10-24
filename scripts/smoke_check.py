import os, re, sys, tempfile, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Isolate from real data and force local (extractive) mode.
os.environ["TRUSTRAG_DATA_DIR"] = tempfile.mkdtemp()
for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
    os.environ[k] = ""

from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.verification.evidence_verifier import verify_claims
from src.verification.reliability import reliability_score
from src.verification.correction_controller import CorrectionController

TEXT = (
    b"Self-Refine uses one model to generate an initial output, critique it, "
    b"and refine it. Across seven tasks, the authors report an average absolute "
    b"improvement of about 20%. Huang et al. (ICLR 2024) find that models struggle "
    b"with intrinsic self-correction in reasoning tasks and performance sometimes "
    b"degrades after the attempt."
)
Q = "What improvement does Self-Refine report?"
BAD = "Across seven tasks, the authors report an average absolute improvement of about 95%."


def new_store():
    s = DocumentStore(tempfile.mkdtemp(), chunk_size=300, chunk_overlap=30)
    return s


def filled_store():
    s = new_store()
    s.add("selfrefine.txt", TEXT)
    return s


class Corrupting:
    def __init__(self, inner): self.inner = inner
    def generate(self, q, ev, instruction=None):
        if instruction is None:
            return re.sub(r"\b20\b", "95", self.inner.generate(q, ev))
        return self.inner.generate(q, ev, instruction=instruction)


class Stubborn:
    def generate(self, q, ev, instruction=None):
        return BAD


class Counting:
    def __init__(self, inner): self.inner, self.calls = inner, 0
    def generate(self, q, ev, instruction=None):
        self.calls += 1
        return self.inner.generate(q, ev)


def t_verifier():
    ev = [{"chunk_id": "x", "text": TEXT.decode()}]
    cases = [
        ("Self-Refine reports an average absolute improvement of about 20%.", "supported"),
        ("Self-Refine reports an average absolute improvement of about 95%.", "insufficient_evidence"),
        ("Intrinsic self-correction improves reasoning-task performance by exactly 95%.", "insufficient_evidence"),
        ("Performance sometimes degrades after intrinsic self-correction.", "supported"),
        ("Self-Refine was evaluated on seven tasks.", "supported"),
        ("Self-Refine was evaluated on twelve tasks.", "insufficient_evidence"),
    ]
    for claim, want in cases:
        r = verify_claims(claim, ev)[0]
        assert r["label"] == want, (claim, r["label"])
        assert r["score"] <= 0.9, ("score not capped", claim)


def t_reliability():
    clean = [{"label": "supported", "negation_warning": False}]
    warned = [{"label": "supported", "negation_warning": True}]
    assert reliability_score(warned, [])["score"] < reliability_score(clean, [])["score"]


def t_controller_fix():
    r = CorrectionController(Retriever(filled_store(), 0), Corrupting(LLMClient()), 2).run(Q, 3, "self_correcting")
    assert r["correction_attempts"][0]["outcome"] == "accepted", r["correction_attempts"]
    assert "20%" in r["final_answer"] and "95%" not in r["final_answer"]
    assert r["answer_status"] == "evidence_supported_by_heuristic"


def t_controller_stubborn():
    r = CorrectionController(Retriever(filled_store(), 0), Stubborn(), 2).run(Q, 3, "self_correcting")
    assert r["answer_status"] == "insufficient_evidence_or_unverified"
    assert "95%" in r["final_answer"]
    assert len(r["correction_attempts"]) == 2, r["correction_attempts"]


def t_no_evidence_no_wasted_call():
    gen = Counting(LLMClient())
    CorrectionController(Retriever(new_store(), 0), gen, 2).run(Q, 3, "self_correcting")
    assert gen.calls == 1, f"expected 1 generator call, got {gen.calls}"


def t_store():
    d = tempfile.mkdtemp()
    s = DocumentStore(d, chunk_size=300, chunk_overlap=30)
    s.add("a.txt", TEXT)
    n = len(s.chunks)
    runtime = Path(d) / "runtime"
    assert not list(runtime.glob("*.tmp")), "temp file left behind"
    assert len(DocumentStore(d, 300, 30).chunks) == n, "reload lost chunks"
    (runtime / "documents.json").write_text("{ broken", encoding="utf-8")
    assert len(DocumentStore(d, 300, 30).chunks) == 0
    assert (runtime / "documents.json.corrupt").exists(), "corrupt file not preserved"


def t_retriever():
    res = Retriever(filled_store(), 0).search(Q, top_k=3)
    assert res and all("similarity" in x for x in res)


def t_api():
    from fastapi.testclient import TestClient
    from src.api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["generation_mode"] == "local_extractive"
    up = c.post("/api/documents/upload", files={"file": ("sr.txt", TEXT, "text/plain")})
    assert up.status_code == 200, up.text
    doc_id = up.json()["document_id"]
    q = c.post("/api/query", json={"question": Q, "top_k": 3, "mode": "self_correcting"})
    assert q.status_code == 200, q.text
    body = q.json()
    for key in ("final_answer", "answer_status", "claims", "correction_attempts", "reliability_score"):
        assert key in body, f"missing key {key}"
    big = c.post("/api/documents/upload", files={"file": ("big.txt", b"a" * (20 * 1024 * 1024 + 1), "text/plain")})
    assert big.status_code == 413, big.status_code
    assert c.delete(f"/api/documents/{doc_id}").status_code == 200


CHECKS = [
    ("verifier labels, numeric guard, score cap", t_verifier),
    ("reliability discounts warned claims", t_reliability),
    ("controller fixes a wrong figure", t_controller_fix),
    ("controller never promotes a bad answer", t_controller_stubborn),
    ("no wasted LLM call when nothing retrieved", t_no_evidence_no_wasted_call),
    ("store: atomic save, reload, corrupt file kept", t_store),
    ("retriever returns scored chunks", t_retriever),
    ("API: health, upload, query, 20MB limit, delete", t_api),
]

failed = 0
for name, fn in CHECKS:
    try:
        fn()
        print("PASS ", name)
    except Exception:
        failed += 1
        print("FAIL ", name)
        traceback.print_exc()

print("\n" + ("ALL CHECKS PASSED" if not failed else f"{failed} CHECK(S) FAILED"))
sys.exit(1 if failed else 0)
