def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def summarize(results):
    total = len(results)
    answered = [r for r in results if not r.get("abstained", False)]
    claim_rows = [c for r in results for c in r.get("claims", []) if c.get("label") != "verification_error"]
    supported = sum(c.get("label") == "supported" for c in claim_rows)
    unsupported = sum(c.get("label") in {"insufficient_evidence", "contradicted"} for c in claim_rows)
    return {
        "questions": total,
        "answer_coverage": round(safe_div(len(answered), total), 4),
        "abstention_rate": round(safe_div(total - len(answered), total), 4),
        "claim_support_rate": round(safe_div(supported, len(claim_rows)), 4),
        "unsupported_claim_rate": round(safe_div(unsupported, len(claim_rows)), 4),
        "mean_latency_seconds": round(safe_div(sum(r.get("latency_seconds", 0) for r in results), total), 4),
        "evaluation_errors": sum(bool(r.get("error")) for r in results)
    }
