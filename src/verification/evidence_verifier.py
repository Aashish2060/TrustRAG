import re
import unicodedata


STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from",
    "are", "was", "were", "has", "have", "had", "its",
    "their", "they", "them", "you", "your", "can", "may",
    "will", "would", "could", "should", "into", "onto",
    "about", "also", "than", "then", "when", "where",
    "which", "who", "what", "how", "why", "each", "both",
    "such", "more", "most", "some", "many", "other",
    "based", "using", "used", "use", "does", "did",
    "notably", "therefore", "however", "overall",
    "these", "those", "there", "here", "section", "table",
    # words that describe the answer/document rather than the facts
    "study", "paper", "document", "evidence",
    "number", "numbers", "value", "values", "reported",
}

# Two-letter words that should never count as acronyms, even in ALL CAPS.
SHORT_STOP_WORDS = {
    "is", "of", "it", "to", "in", "on", "an", "as", "at", "be",
    "by", "if", "or", "we", "he", "so", "do", "no", "up", "us",
    "my", "me", "am", "go",
}

# Heuristic scores are never reported as certain.
HEURISTIC_MAX_SCORE = 0.9

# Citation markers: [chunk_id], [id1, id2], [1], [1-3], and 【chunk_id】
CITATION_PATTERN = re.compile(
    r"\[(?:[A-Za-z0-9_-]{6,}(?:\s*[,;]\s*[A-Za-z0-9_-]{6,})*"
    r"|\d+(?:\s*[,;-]\s*\d+)*)\]"
    r"|【[^】]*】"
)

# Matches 20, 3.5, 1,000, 1,000.5 (the % sign is simply ignored).
NUMBER_PATTERN = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")

# "one" is deliberately excluded: it is usually a pronoun ("one model").
WORD_NUMBERS = {
    "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000",
}

# Stem groups with opposite meaning. Used only for a warning flag.
POSITIVE_STEMS = (
    "improv", "gain", "increas", "boost", "enhanc", "outperform",
)
NEGATIVE_STEMS = (
    "degrad", "declin", "decreas", "worsen", "drop", "reduc",
    "underperform",
)

ABBREVIATIONS = {
    "et al.": "ET_AL_TOKEN",
    "e.g.": "EG_TOKEN",
    "i.e.": "IE_TOKEN",
    "vs.": "VS_TOKEN",
    "Fig.": "FIG_TOKEN",
    "Dr.": "DR_TOKEN",
    "Inc.": "INC_TOKEN",
    "approx.": "APPROX_TOKEN",
    "Eq.": "EQ_TOKEN",
}

# Typographic characters that LLMs often emit, mapped to plain ASCII.
_CHAR_MAP = {
    0x200B: None, 0x200C: None, 0x200D: None, 0x2060: None, 0xFEFF: None,  # invisible
    0x2010: "-", 0x2011: "-",                       # hyphens
    0x2012: " - ", 0x2013: " - ", 0x2014: " - ",    # dashes
    0x2015: " - ", 0x2212: " - ",
    0x2018: "'", 0x2019: "'",                       # single quotes
    0x201C: '"', 0x201D: '"',                       # double quotes
}

# British -> American spellings, so "defences" matches "defenses".
# Applied to both claim and evidence, so matching stays consistent.
_UK_US = {
    "defence": "defense", "defences": "defenses",
    "behaviour": "behavior", "behaviours": "behaviors",
    "colour": "color", "colours": "colors",
    "centre": "center", "centres": "centers",
    "analyse": "analyze", "analysed": "analyzed",
    "labelled": "labeled", "modelling": "modeling",
    "favour": "favor", "programme": "program",
    "licence": "license", "grey": "gray",
}
_UK_SUFFIXES = (
    ("isations", "izations"), ("isation", "ization"),
    ("ising", "izing"), ("ised", "ized"),
    ("ises", "izes"), ("ise", "ize"),
)

# Sentences where the answer declines to answer (not factual claims).
ABSTENTION_PATTERNS = [
    re.compile(
        r"\b(?:evidence|documents?|sources?|passages?|context|text)\b"
        r".{0,60}?\b(?:does not|do not|doesn't|don't|did not|not)\b"
        r".{0,40}?\b(?:contain|provide|include|mention|specify|state|"
        r"address|say|cover|have|describe)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i|we)\s+(?:cannot|can't|can not|am unable to|"
        r"am not able to)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i'm|i am)\s+sorry\b", re.IGNORECASE),
    re.compile(
        r"\bno\s+(?:relevant\s+|sufficient\s+|sufficiently relevant\s+)?"
        r"(?:information|evidence)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:insufficient|not enough)\s+(?:evidence|information)\b",
        re.IGNORECASE,
    ),
]

# Sentences that only describe the evidence, e.g. "The evidence is mixed."
META_PATTERN = re.compile(
    r"^(?:the\s+)?(?:supplied\s+|provided\s+|available\s+|retrieved\s+)?"
    r"(?:evidence|documents?|sources?|passages?)\s+(?:is|are)\s+"
    r"(?:mixed|limited|inconclusive|contradictory|conflicting|"
    r"insufficient)\b",
    re.IGNORECASE,
)

# "The document lists several limitations ..." (framing, not a fact)
DOC_DESCRIBES_PATTERN = re.compile(
    r"^(?:the\s+)?(?:document|paper|study|text|evidence|passage|source)s?\s+"
    r"(?:lists|cites|provides|presents|outlines|mentions|highlights)\s+"
    r"(?:several|a\s+(?:set|number|few|range|list)|the\s+following|"
    r"some|various|multiple)\b",
    re.IGNORECASE,
)

# "These counts are listed in the ... table of the evidence."
POINTER_PATTERN = re.compile(
    r"^(?:these|those|this|they|all)\b.{0,40}?\b(?:are|were|is)\s+"
    r"(?:also\s+)?(?:listed|shown|reported|drawn|taken|found|given|"
    r"mentioned|cited|stated|based)\b.{0,25}?\b(?:in|from|on)\b",
    re.IGNORECASE,
)

VAGUE_PATTERN = re.compile(r"\bone of several\b", re.IGNORECASE)

# Bullet or numbered list item.
BULLET_PATTERN = re.compile(r"^\s*(?:[-*•▪●◦]|\d{1,2}[.)])\s+(.+)$")

# Table cells that carry no fact ("-", "n/a", "(no GPU value reported)").
PLACEHOLDER_CELL = re.compile(
    r"^[-\s]*\(?(?:n/?a|none|null|"
    r"not\s+(?:reported|available|applicable|given)|"
    r"no\b[^)]{0,40}(?:reported|available|given|value)[^)]*)\)?$",
    re.IGNORECASE,
)

# Header-ish first cells used only for tables without a separator row.
HEADER_FIRST_CELLS = {
    "category", "claim", "statement", "topic", "item",
    "type", "method", "model", "dataset", "metric",
}


def _clean_text(text: str) -> str:
    """Normalize Unicode, remove citation markers and formatting."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_CHAR_MAP)
    text = CITATION_PATTERN.sub("", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def _stem(word: str) -> str:
    """Very light plural stemming: tasks -> task, degrades -> degrade."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith(("ss", "us", "is"))
    ):
        return word[:-1]
    return word


def _us_spelling(word: str) -> str:
    """Map common British spellings to American ones."""
    if word in _UK_US:
        return _UK_US[word]
    for uk, us in _UK_SUFFIXES:
        if word.endswith(uk) and len(word) > len(uk) + 3:
            return word[: -len(uk)] + us
    return word


def _tokens(text: str) -> set[str]:
    """Extract normalized terms for approximate lexical matching.

    Hyphenated words are split ("deep-learning" -> deep, learning) so they
    match the same words written with a space.
    """
    cleaned = _clean_text(text)

    raw_tokens = re.findall(r"[A-Za-z0-9]+", cleaned)

    result = set()
    for raw in raw_tokens:
        low = raw.lower()

        if low in STOP_WORDS:
            continue

        if len(low) > 2:
            result.add(_stem(_us_spelling(low)))
        elif (
            len(raw) == 2
            and raw.isalpha()
            and raw.isupper()
            and low not in SHORT_STOP_WORDS
        ):
            # Keep short acronyms such as AI, RL, QA, AP.
            result.add(low)

    return result


def _normalize_number(value: str) -> str:
    """Normalize '1,000' -> '1000', '20.0' -> '20', '3.50' -> '3.5'."""
    value = value.replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return value

    if number == int(number):
        return str(int(number))
    return repr(number)


def _numbers(text: str) -> set[str]:
    """Extract normalized numeric values (digits and number words)."""
    cleaned = _clean_text(text)

    found = {
        _normalize_number(match)
        for match in NUMBER_PATTERN.findall(cleaned)
    }

    for word in re.findall(r"[a-z]+", cleaned.lower()):
        if word in WORD_NUMBERS:
            found.add(WORD_NUMBERS[word])

    return found


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences while protecting common abbreviations."""
    text = _clean_text(text)

    for original, token in ABBREVIATIONS.items():
        text = re.sub(
            r"(?<![A-Za-z])" + re.escape(original),
            token,
            text,
            flags=re.IGNORECASE,
        )

    sentences = re.split(r"(?<=[.!?])\s+", text)

    restored = []
    for sentence in sentences:
        for original, token in ABBREVIATIONS.items():
            sentence = sentence.replace(token, original)

        sentence = _clean_text(sentence)
        if sentence:
            restored.append(sentence)

    return restored


def _is_abstention_sentence(sentence: str) -> bool:
    """True for 'the evidence does not say...' style sentences.

    Sentences containing figures are never skipped, so an unsupported
    number cannot hide inside an abstention-looking sentence.
    """
    if _numbers(sentence):
        return False
    return any(p.search(sentence) for p in ABSTENTION_PATTERNS)


def _is_non_claim(sentence: str) -> bool:
    """Abstentions, intro lines and remarks about the evidence are not
    factual claims. Sentences containing figures are always checked."""
    if _is_abstention_sentence(sentence):
        return True

    if _numbers(sentence):
        return False

    if sentence.rstrip().endswith(":"):
        return True

    return bool(
        META_PATTERN.match(sentence)
        or DOC_DESCRIBES_PATTERN.match(sentence)
        or POINTER_PATTERN.match(sentence)
        or VAGUE_PATTERN.search(sentence)
    )


def is_abstention(answer: str) -> bool:
    """True if every sentence of the answer declines to answer."""
    sentences = _split_sentences(answer or "")
    return bool(sentences) and all(
        _is_abstention_sentence(s) for s in sentences
    )


def _is_placeholder(cell: str) -> bool:
    cell = (cell or "").strip()
    if not re.search(r"[A-Za-z0-9]", cell):
        return True
    return bool(PLACEHOLDER_CELL.match(cell))


def _table_entries(lines: list[str]) -> list[tuple[str, str | None]]:
    """Turn a markdown table into one claim per data cell.

    The claim is "<row label> <column header>: <cell>", for example
    "RetinaFace FPS (CPU): 4 FPS". The second item is a fallback claim
    without the column header, "RetinaFace: 4 FPS", used when the header
    sits in a different passage than the row.
    """
    rows = []
    sep_idx = None

    for line in lines:
        raw_cells = [c.strip() for c in line.strip().strip("|").split("|")]

        is_separator = bool(raw_cells) and all(
            re.fullmatch(r":?-{3,}:?", c.replace(" ", ""))
            for c in raw_cells
        )
        if is_separator:
            if sep_idx is None:
                sep_idx = len(rows)
            continue

        rows.append([_clean_text(c) for c in raw_cells])

    header = None
    if sep_idx is not None and sep_idx >= 1:
        header = rows[sep_idx - 1]
        data = rows[: sep_idx - 1] + rows[sep_idx:]
    else:
        data = [
            r for r in rows
            if not (r and r[0].strip().lower().rstrip(":") in HEADER_FIRST_CELLS)
        ]

    entries: list[tuple[str, str | None]] = []

    for row in data:
        if not any(row):
            continue

        label = row[0] if row and not _is_placeholder(row[0]) else ""

        if len(row) == 1:
            if label:
                entries.append((label, None))
            continue

        for j, cell in enumerate(row[1:], start=1):
            if not cell or _is_placeholder(cell):
                continue

            column = header[j] if header and j < len(header) else ""
            core = f"{label}: {cell}" if label else cell

            if column and label:
                full = f"{label} {column}: {cell}"
                entries.append((full, core))
            else:
                entries.append((core, None))

    return entries


def _split_claims_detailed(answer: str) -> list[tuple[str, str | None]]:
    """Split an answer into (claim, fallback_claim) pairs.

    - Prose becomes sentences.
    - Every bullet or numbered item becomes its own claim.
    - Short headings and lines ending with ':' that introduce a list are
      dropped (they are labels, not facts).
    - Markdown tables become one claim per data cell, using the header.
    - Abstentions, framing sentences and duplicates are removed.
    """
    if not answer or not answer.strip():
        return []

    lines = [ln.rstrip() for ln in answer.splitlines()]
    entries: list[tuple[str, str | None]] = []
    prose: list[str] = []
    table_block: list[str] = []

    def flush_prose():
        if prose:
            text = " ".join(prose)
            prose.clear()
            for sentence in _split_sentences(text):
                entries.append((sentence, None))

    def flush_table():
        if table_block:
            entries.extend(_table_entries(table_block))
            table_block.clear()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            flush_prose()
            flush_table()
            continue

        if "|" in stripped:
            flush_prose()
            table_block.append(stripped)
            continue

        flush_table()

        bullet = BULLET_PATTERN.match(stripped)
        if bullet:
            flush_prose()
            for sentence in _split_sentences(bullet.group(1)):
                entries.append((sentence, None))
            continue

        plain = _clean_text(stripped)
        next_line = next(
            (l.strip() for l in lines[i + 1:] if l.strip()), ""
        )
        next_is_list = bool(BULLET_PATTERN.match(next_line)) or "|" in next_line

        if next_is_list and (
            plain.endswith(":")
            or (len(plain.split()) <= 8 and not re.search(r"[.!?]$", plain))
        ):
            flush_prose()
            continue  # heading or intro line for the list below

        prose.append(stripped)

    flush_prose()
    flush_table()

    unique: list[tuple[str, str | None]] = []
    seen = set()

    for claim, fallback in entries:
        normalized = _clean_text(claim)

        if not _tokens(normalized):
            continue
        if _is_non_claim(normalized):
            continue

        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)

        fb = _clean_text(fallback) if fallback else None
        unique.append((normalized, fb))

    return unique


def split_claims(answer: str) -> list[str]:
    """Split an answer into individually checkable claims."""
    return [claim for claim, _ in _split_claims_detailed(answer)]


def _has_negation(text: str) -> bool:
    """Detect basic negation for diagnostics, not contradiction decisions."""
    return bool(
        re.search(
            r"\b(?:not|no|never|none|without|cannot|can't|"
            r"doesn't|isn't|aren't|neither|nor|false)\b|n['’]t\b",
            text.lower(),
        )
    )


def _direction(text: str) -> tuple[bool, bool]:
    """Return (has_positive_term, has_negative_term) using word stems."""
    words = re.findall(r"[a-z]+", text.lower())
    positive = any(w.startswith(POSITIVE_STEMS) for w in words)
    negative = any(w.startswith(NEGATIVE_STEMS) for w in words)
    return positive, negative


def _has_negation_conflict(claim: str, evidence_text: str) -> bool:
    """
    Conservatively identify possible polarity differences.

    Two signals are used, both warning-only (they never change the label):
      1. One side is negated (not/never/...) and the other is not.
      2. One side uses an "improve"-type term and the other a
         "degrade"-type term.
    """
    claim_tokens = _tokens(claim)
    if len(claim_tokens) < 3:
        return False

    claim_negated = _has_negation(claim)
    claim_pos, claim_neg = _direction(claim)

    for sentence in _split_sentences(evidence_text):
        sentence_tokens = _tokens(sentence)
        shared = claim_tokens & sentence_tokens

        overlap = len(shared) / max(1, len(claim_tokens))

        if len(shared) < 3 or overlap < 0.40:
            continue

        if overlap >= 0.60 and claim_negated != _has_negation(sentence):
            return True

        sent_pos, sent_neg = _direction(sentence)

        if claim_pos and not claim_neg and sent_neg and not sent_pos:
            return True
        if claim_neg and not claim_pos and sent_pos and not sent_neg:
            return True

    return False


def _assess_claim(claim: str, usable_evidence: list[dict]) -> dict:
    """Score one claim against the retrieved passages (lexical heuristic)."""
    claim_tokens = _tokens(claim)

    if not claim_tokens or not usable_evidence:
        return {
            "claim": claim,
            "label": "insufficient_evidence",
            "evidence_ids": [],
            "score": 0.0,
            "negation_warning": False,
            "missing_numbers": [],
            "explanation": "No usable claim tokens or retrieved evidence.",
        }

    scored = []
    for row in usable_evidence:
        overlap = len(claim_tokens & row["tokens"]) / len(claim_tokens)
        scored.append({
            "score": overlap,
            "item": row["item"],
            "text": row["text"],
            "tokens": row["tokens"],
        })

    scored.sort(key=lambda row: row["score"], reverse=True)
    best = scored[0]

    supporting_rows = [row for row in scored if row["score"] >= 0.20][:3]

    combined_tokens = set()
    for row in supporting_rows:
        combined_tokens.update(row["tokens"])

    combined_score = len(claim_tokens & combined_tokens) / len(claim_tokens)

    lexically_supported = (
        best["score"] >= 0.60
        or (
            combined_score >= 0.70
            and len(supporting_rows) >= 2
            and best["score"] >= 0.30
        )
    )

    # Numeric guard: every figure in the claim must appear in the
    # supporting passages (or the best passage if none qualify).
    claim_numbers = _numbers(claim)
    number_rows = supporting_rows or [best]
    evidence_numbers = set()
    for row in number_rows:
        evidence_numbers |= _numbers(row["text"])

    missing_numbers = sorted(claim_numbers - evidence_numbers)
    supported = lexically_supported and not missing_numbers

    negation_warning = any(
        _has_negation_conflict(claim, row["text"])
        for row in supporting_rows
    )

    if supported:
        label = "supported"
        explanation = (
            "Strong lexical overlap with retrieved evidence. "
            "This heuristic does not prove semantic entailment."
        )
        relevant_rows = (
            [best] if best["score"] >= 0.60 else supporting_rows
        )
        score = max(
            best["score"],
            combined_score if len(relevant_rows) > 1 else 0.0,
        )
    else:
        label = "insufficient_evidence"
        if lexically_supported and missing_numbers:
            explanation = (
                "Wording overlaps with the evidence, but the claim "
                "contains figures not found in it: "
                + ", ".join(missing_numbers) + "."
            )
        else:
            explanation = (
                "Lexical overlap is insufficient to establish support."
            )
        relevant_rows = [best] if best["score"] >= 0.25 else []
        score = best["score"]

    score = min(score, HEURISTIC_MAX_SCORE)

    if negation_warning:
        explanation += (
            " A possible polarity difference was detected; "
            "manual or semantic verification is recommended."
        )

    evidence_ids = list(dict.fromkeys(
        row["item"].get("chunk_id")
        for row in relevant_rows
        if row["item"].get("chunk_id")
    ))

    return {
        "claim": claim,
        "label": label,
        "evidence_ids": evidence_ids,
        "score": round(score, 3),
        "negation_warning": negation_warning,
        "missing_numbers": missing_numbers,
        "explanation": explanation,
    }


def verify_claims(answer: str, evidence: list[dict]) -> list[dict]:
    """
    Assess claims with lexical overlap against retrieved passages.

    Labels are heuristic:
      - supported: high lexical overlap AND every number in the claim
        appears in the supporting evidence
      - insufficient_evidence: evidence is weak, ambiguous, or the claim
        contains figures not found in the evidence

    For table cells, the claim includes the column header. If that fails
    only because the header is in another passage, the row label plus the
    value is checked instead, and the result is flagged
    "matched_without_column".

    This does not establish semantic entailment, and it cannot detect a
    wrong column (e.g. a recall value described as precision) when both
    words and the number appear in the evidence.
    """
    usable_evidence = []

    for item in evidence or []:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "") or "")
        tokens = _tokens(text)

        if tokens:
            usable_evidence.append({
                "item": item,
                "text": text,
                "tokens": tokens,
            })

    results = []

    for claim, fallback in _split_claims_detailed(answer):
        result = _assess_claim(claim, usable_evidence)

        if result["label"] != "supported" and fallback:
            alt = _assess_claim(fallback, usable_evidence)
            if alt["label"] == "supported":
                alt["claim"] = claim
                alt["matched_without_column"] = True
                alt["explanation"] += (
                    " Matched on row label and value; the column header "
                    "was not found in the same passage."
                )
                result = alt

        results.append(result)

    return results