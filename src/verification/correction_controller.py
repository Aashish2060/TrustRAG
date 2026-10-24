import re
import time
from itertools import permutations

from src.verification.evidence_verifier import verify_claims, is_abstention
from src.verification.reliability import reliability_score


# ----------------------------------------------------------------------
# Rules given to the generator (used on revisions and, if enabled, on the
# first answer too)
# ----------------------------------------------------------------------
USE_RULES_ON_FIRST_ANSWER = True

_BASE_RULES = (
    "Use only facts and numbers that appear in the evidence. "
    "Do not compute ratios, percentages, differences or 'times faster/slower'. "
    "Do not write introductory or concluding sentences such as "
    "'The document lists...' or 'These points are drawn from...'. "
    "State each fact directly. Prefer a short answer. "
    "If the evidence is insufficient, explicitly abstain."
)


# ----------------------------------------------------------------------
# Filter: sentences that are framing, not factual claims
# ----------------------------------------------------------------------
_META = re.compile(
    r"^(these (points|counts|figures|results|numbers|details) (are|were) "
    r"(drawn|listed|taken|from|shown|reported)\b"
    r"|the (document|evidence|paper|study) (lists|cites|provides) "
    r"(several|a set|the following|a number)\b"
    r"|according to the (evidence|document|paper)[,:]?\s*$)",
    re.I,
)
_VAGUE = re.compile(r"\bone of several\b", re.I)


def _is_checkable(claim: dict) -> bool:
    text = (claim.get("claim") or "").strip()
    if not text:
        return False
    if text.endswith(":"):  # intro line like "The study used four datasets:"
        return False
    if _META.match(text) or _VAGUE.search(text):
        return False
    return True


# ----------------------------------------------------------------------
# Derived numbers: accept a figure if it is simple arithmetic on figures
# that really are in the evidence (e.g. 30 / 6 = 5, 30 / 4 = 7.5)
# ----------------------------------------------------------------------
_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _evidence_text(e) -> str:
    if isinstance(e, dict):
        for key in ("text", "content", "chunk", "page_content"):
            if e.get(key):
                return str(e[key])
        return ""
    return str(e)


def _numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def _to_float(s) -> float | None:
    try:
        return float(str(s).replace(",", "").strip().rstrip("."))
    except ValueError:
        return None


def _is_derivable(target: float, pool: list[float]) -> bool:
    pool = sorted(set(pool))[:60]  # keep it cheap
    tol = max(0.011, abs(target) * 0.01)
    for a, b in permutations(pool, 2):
        if b != 0 and abs(a / b - target) <= tol:
            return True
        if abs(a - b - target) <= tol or abs(a + b - target) <= tol:
            return True
        if b != 0 and abs((a - b) / b * 100 - target) <= tol:  # percent change
            return True
    return False


def _apply_derived_numbers(claims: list[dict], evidence: list) -> list[dict]:
    """Relabel a claim as supported when its only problem is figures that
    are derivable from the evidence, and its score is otherwise decent."""
    ev_numbers = _numbers_in(" ".join(_evidence_text(e) for e in evidence))
    if not ev_numbers:
        return claims

    result = []
    for c in claims:
        missing = c.get("missing_numbers") or []
        score = c.get("score")
        if (
            c.get("label") != "supported"
            and missing
            and (score is None or score >= 0.5)
        ):
            vals = [_to_float(m) for m in missing]
            if all(v is not None and _is_derivable(v, ev_numbers) for v in vals):
                c = {
                    **c,
                    "label": "supported",
                    "missing_numbers": [],
                    "derived_numbers": list(missing),
                }
        result.append(c)
    return result


def _verify(answer: str, evidence: list) -> tuple[list[dict], int]:
    """Verify, drop framing sentences, accept derivable numbers.
    Returns (claims, number_of_filtered_framing_claims)."""
    raw = verify_claims(answer, evidence)
    kept = [c for c in raw if _is_checkable(c)]
    # Safety net: never filter away everything (that would hide real claims).
    if not kept and raw and not is_abstention(answer):
        kept = raw
    filtered = len(raw) - len(kept)
    return _apply_derived_numbers(kept, evidence), filtered


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _supported_count(claims: list[dict]) -> int:
    return sum(c["label"] == "supported" for c in claims)


def _supported_ratio(claims: list[dict]) -> float:
    return _supported_count(claims) / len(claims) if claims else 0.0


def _build_revision_instruction(failed_claims: list[dict]) -> str:
    lines = [
        "Correct only claims that are not supported.",
        _BASE_RULES,
    ]
    if failed_claims:
        lines.append("These claims could not be verified against the evidence:")
        for claim in failed_claims:
            line = f"- {claim['claim']}"
            missing = claim.get("missing_numbers") or []
            if missing:
                line += (
                    " (figures not found in the evidence: "
                    + ", ".join(missing)
                    + "; correct or remove them)"
                )
            lines.append(line)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------
class CorrectionController:
    def __init__(self, retriever, generator, max_corrections=2):
        self.retriever, self.generator = retriever, generator
        self.max_corrections = max(0, min(2, int(max_corrections)))

    def run(self, question: str, top_k: int = 4, mode: str = "self_correcting"):
        start = time.perf_counter()

        evidence = self.retriever.search(question, top_k=top_k)

        if USE_RULES_ON_FIRST_ANSWER and evidence:
            try:
                initial = self.generator.generate(
                    question, evidence, instruction=_BASE_RULES
                )
            except TypeError:  # generator without instruction support
                initial = self.generator.generate(question, evidence)
        else:
            initial = self.generator.generate(question, evidence)

        initial_claims, filtered_total = _verify(initial, evidence)

        attempts = []
        final_answer = initial
        final_claims = initial_claims
        final_evidence = evidence

        pool = list(evidence)
        tried_without_new_evidence = False

        if mode == "self_correcting":
            for attempt_num in range(self.max_corrections):
                failed = [c for c in final_claims if c["label"] != "supported"]
                if not failed:
                    break

                seen = {e["chunk_id"] for e in pool}
                expanded = self.retriever.search(
                    question,
                    top_k=max(top_k * 2, top_k + 2),
                    exclude_ids=seen,
                )

                if not expanded:
                    if tried_without_new_evidence or not pool:
                        attempts.append({
                            "attempt": attempt_num + 1,
                            "outcome": "stopped_no_new_evidence",
                            "new_evidence_count": 0,
                        })
                        break
                    tried_without_new_evidence = True
                else:
                    pool.extend(expanded)

                revised = self.generator.generate(
                    question,
                    pool,
                    instruction=_build_revision_instruction(failed),
                )
                revised_claims, revised_filtered = _verify(revised, pool)

                before = _supported_count(final_claims)
                after = _supported_count(revised_claims)
                all_ok = bool(revised_claims) and all(
                    c["label"] == "supported" for c in revised_claims
                )
                # Accept a perfect revision, or one that is clearly better
                # (higher share of supported claims, nothing lost).
                better = (
                    bool(revised_claims)
                    and _supported_ratio(revised_claims)
                    > _supported_ratio(final_claims)
                    and after >= min(before, len(revised_claims))
                )
                accepted = all_ok or better

                attempts.append({
                    "attempt": attempt_num + 1,
                    "outcome": "accepted" if accepted else "rejected_or_unresolved",
                    "new_evidence_count": len(expanded),
                    "supported_claims_before": before,
                    "supported_claims_after": after,
                    "supported_ratio_before": round(_supported_ratio(final_claims), 3),
                    "supported_ratio_after": round(_supported_ratio(revised_claims), 3),
                })

                if accepted:
                    final_answer = revised
                    final_claims = revised_claims
                    final_evidence = list(pool)
                    filtered_total = revised_filtered
                    if all_ok:
                        break
                # Otherwise keep the previous answer, claims and evidence
                # as a consistent set, and try again.

        reliability = reliability_score(final_claims, final_evidence)

        if (
            not final_evidence
            or not final_claims
            or any(c["label"] != "supported" for c in final_claims)
        ):
            status = "insufficient_evidence_or_unverified"
        else:
            status = "evidence_supported_by_heuristic"

        return {
            "question": question,
            "initial_answer": initial,
            "final_answer": final_answer,
            "answer_status": status,
            "claims": final_claims,
            "evidence": final_evidence,
            "correction_attempts": attempts,
            "reliability_score": reliability["score"],
            "reliability_signals": reliability,
            "framing_sentences_ignored": filtered_total,
            "latency_seconds": round(time.perf_counter() - start, 4),
            "abstained": (not final_claims) and is_abstention(final_answer),
            "mode": mode,
        }