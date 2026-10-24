from src.verification.evidence_verifier import split_claims, verify_claims
from src.verification.reliability import reliability_score

def test_claim_split():
    assert len(split_claims("First claim. Second claim.")) == 2

def test_matching_evidence_supported_heuristic():
    evidence = [{"chunk_id":"c1", "text":"Solar panels convert sunlight into electrical energy using photovoltaic cells.", "similarity":0.7}]
    result = verify_claims("Solar panels convert sunlight into electrical energy using photovoltaic cells. [c1]", evidence)
    assert result[0]["label"] == "supported"

def test_no_evidence_is_insufficient():
    result = verify_claims("The capital is Kathmandu.", [])
    assert result[0]["label"] == "insufficient_evidence"

def test_reliability_is_bounded():
    score = reliability_score([{"label":"supported"}], [{"similarity":0.9}])["score"]
    assert 0 <= score <= 1
