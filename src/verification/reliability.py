import math


def reliability_score(claims: list[dict], evidence: list[dict]) -> dict:
    """
    Calculate a transparent heuristic reliability score.

    This score is NOT a probability that an answer is correct.
    Retrieval similarity alone does not establish factual support.
    """

    valid_claims = [
        claim
        for claim in claims
        if claim.get("label") != "verification_error"
    ]

    total = len(valid_claims)

    supported = sum(
        claim.get("label") == "supported"
        for claim in valid_claims
    )

    contradicted = sum(
        claim.get("label") == "contradicted"
        for claim in valid_claims
    )

    insufficient = sum(
        claim.get("label") == "insufficient_evidence"
        for claim in valid_claims
    )

    if total == 0:
        return {
            "score": 0.0,
            "supported_fraction": 0.0,
            "contradicted_fraction": 0.0,
            "insufficient_fraction": 0.0,
            "retrieval_signal": 0.0,
            "claim_count": 0,
            "supported_claims": 0,
            "contradicted_claims": 0,
            "insufficient_claims": 0,
            "warned_supported_claims": 0,
            "score_type": "heuristic_not_probability",
        }

    supported_fraction = supported / total
    contradicted_fraction = contradicted / total
    insufficient_fraction = insufficient / total

    # Supported claims that also carry a polarity warning count as half.
    warned = sum(
        claim.get("label") == "supported"
        and bool(claim.get("negation_warning", False))
        for claim in valid_claims
    )
    effective_supported_fraction = (supported - 0.5 * warned) / total

    # Ignore absent, invalid, or non-finite similarity values.
    similarities = []

    for item in evidence:
        raw_similarity = item.get("similarity")

        if raw_similarity is None:
            continue

        try:
            similarity = float(raw_similarity)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(similarity):
            continue

        similarities.append(max(0.0, min(1.0, similarity)))

    retrieval_signal = (
        sum(similarities) / len(similarities)
        if similarities
        else 0.0
    )

    # Retrieval similarity can contribute only when at least one
    # claim has been labelled supported by the verifier.
    retrieval_contribution = (
        0.30 * retrieval_signal
        if supported > 0
        else 0.0
    )

    score = (
        0.70 * effective_supported_fraction
        + retrieval_contribution
        - 0.50 * contradicted_fraction
    )

    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "supported_fraction": round(supported_fraction, 3),
        "contradicted_fraction": round(contradicted_fraction, 3),
        "insufficient_fraction": round(insufficient_fraction, 3),
        "retrieval_signal": round(retrieval_signal, 3),
        "claim_count": total,
        "supported_claims": supported,
        "contradicted_claims": contradicted,
        "insufficient_claims": insufficient,
        "warned_supported_claims": warned,
        "score_type": "heuristic_not_probability",
    }