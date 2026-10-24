from pydantic import BaseModel, Field
from typing import Literal

VerificationLabel = Literal["supported", "contradicted", "insufficient_evidence", "verification_error"]

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    mode: Literal["baseline", "verified", "self_correcting"] = "self_correcting"

class ClaimCheck(BaseModel):
    claim: str
    label: VerificationLabel
    evidence_ids: list[str] = []
    score: float = 0.0
    explanation: str

class QueryResponse(BaseModel):
    question: str
    initial_answer: str
    final_answer: str
    answer_status: str
    claims: list[ClaimCheck]
    evidence: list[dict]
    correction_attempts: list[dict]
    reliability_score: float
    latency_seconds: float
    mode: str
