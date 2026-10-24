import math
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Question words that say nothing about the content being looked for.
FILLER = {
    "achieve", "achieved", "achieves", "higher", "lower", "highest",
    "lowest", "best", "worst", "value", "values", "mentioned", "used",
    "use", "many", "much", "get", "got", "paper", "study", "document",
    "tell", "describe", "explain", "give", "list", "does", "did", "which",
    "proposed",
    # added: words that only frame the question
    "say", "said", "state", "claim", "report", "mention", "show",
    "according", "authors",
    # added: inflected forms the stemmer doesn't reduce, plus framing words
    "reported", "found", "including",
}

# Question word -> words the document may use instead.
SYNONYMS = {
    "author": ["submitted", "by", "names", "students", "written"],
    "accuracy": ["precision", "recall", "ap", "performance"],
    "speed": ["fps", "frames"],
    "fast": ["fps"],
    "faster": ["fps"],
    "fastest": ["fps"],
    "slow": ["fps"],
    "slower": ["fps"],
    "slowest": ["fps"],
    "compare": ["comparison", "versus", "performance"],
    "limitation": ["challenges", "bias", "drawbacks"],
    "future": ["scope", "enhancements", "development"],
    "dataset": ["datasets", "images", "faces"],
    "image": ["images"],
    "conclusion": ["results", "show", "outperform"],
    "conclude": ["results", "show", "outperform"],
    "university": ["department", "roorkee"],
}

# Splits a long multi-part question into sub-questions.
SPLIT_RE = re.compile(
    r"\s*(?:,\s*(?:including|and)?|;|\band\b|\bas well as\b|\bincluding\b)\s*",
    re.IGNORECASE,
)


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _terms(text: str) -> list[str]:
    return [
        _stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 1 and t not in ENGLISH_STOP_WORDS
    ]


class Retriever:
    def __init__(self, store, min_relevance=0.03, min_coverage=0.6):
        self.store = store
        # Never stricter than 0.03: the vocabulary fix below already
        # removes the dilution that made real matches score very low.
        self.min_relevance = min(min_relevance, 0.03)
        # Share of the question's (weighted) content words that must be
        # found in a chunk. This is what keeps unanswerable questions
        # (e.g. "What GPU was used for training?") from retrieving junk.
        # If real questions abstain, try 0.5. If trap questions start
        # answering, try 0.7.
        self.min_coverage = min_coverage

    def _query_groups(self, query: str) -> list[list[str]]:
        """One group per content word: the word plus its synonyms."""
        groups, seen = [], set()
        for t in _terms(query):
            if t in FILLER or t in seen:
                continue
            seen.add(t)
            groups.append([t] + [_stem(s) for s in SYNONYMS.get(t, [])])
        return groups

    def search(
        self,
        query: str,
        top_k: int = 4,
        exclude_ids: set[str] | None = None,
    ) -> list[dict]:
        excluded = exclude_ids or set()

        chunks = [
            c for c in self.store.chunks
            if c.get("chunk_id") not in excluded
        ]
        if not chunks:
            return []

        texts = [c.get("text", "") for c in chunks]
        groups = self._query_groups(query)
        if not groups:
            return []

        # Query text for TF-IDF: content words plus synonyms.
        expanded_query = " ".join(w for g in groups for w in g)

        # Fit on the chunks ONLY. Words that appear only in the question
        # no longer inflate the query vector.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
            qvec = vectorizer.transform([expanded_query])
            cosine = cosine_similarity(qvec, matrix).ravel()
        except ValueError:
            return []

        # Weighted coverage of the question's content words per chunk.
        chunk_terms = [set(_terms(t)) for t in texts]
        n = len(chunks)
        df = {}
        for terms in chunk_terms:
            for t in terms:
                df[t] = df.get(t, 0) + 1
        max_idf = math.log(n + 1) + 1

        def idf(word):
            # A word found nowhere in the document is the most informative
            # (and counts fully against coverage).
            return math.log((n + 1) / (df[word] + 1)) + 1 if word in df else max_idf

        group_weight = [max(idf(w) for w in g) for g in groups]
        total_weight = sum(group_weight)

        coverage = np.zeros(n)
        for i, terms in enumerate(chunk_terms):
            got = sum(
                wt for g, wt in zip(groups, group_weight)
                if any(w in terms for w in g)
            )
            coverage[i] = got / total_weight if total_weight else 0.0

        scores = cosine.copy()

        # Mild boost when a chunk's section title shares a real query word.
        query_terms = {g[0] for g in groups}
        for i, chunk in enumerate(chunks):
            heading = {
                _stem(w) for w in re.findall(
                    r"[a-z0-9]+", (chunk.get("section_title") or "").lower()
                )
            }
            if query_terms & heading:
                scores[i] *= 1.2

        # Rank by similarity, favouring chunks that cover more of the question.
        ranking = scores * (0.5 + coverage)
        order = np.argsort(ranking)[::-1]

        results, seen_texts = [], set()
        for idx in order:
            score = float(scores[idx])

            if score < self.min_relevance or coverage[idx] < self.min_coverage:
                continue

            item = dict(chunks[int(idx)])
            normalized = re.sub(r"\s+", " ", item.get("text", "").strip().lower())
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            item["similarity"] = round(score, 4)
            item["coverage"] = round(float(coverage[idx]), 3)
            results.append(item)

            if len(results) >= top_k:
                break

        return results

    def search_compound(
        self,
        query: str,
        top_k: int = 4,
        exclude_ids: set[str] | None = None,
        max_groups: int = 6,
    ) -> list[dict]:
        """Split long multi-part questions, search each part, merge results.

        Short questions go straight to search(), so behaviour for them is
        unchanged. Each merged item records which sub-question it matched in
        `matched_subquery`.
        """
        if len(self._query_groups(query)) <= max_groups:
            return self.search(query, top_k, exclude_ids)

        parts = [p.strip() for p in SPLIT_RE.split(query) if p and p.strip()]
        parts = [p for p in parts if len(self._query_groups(p)) >= 2]
        if len(parts) < 2:
            return self.search(query, top_k, exclude_ids)

        merged = {}
        for part in parts:
            for item in self.search(part, top_k=top_k, exclude_ids=exclude_ids):
                cid = item["chunk_id"]
                if cid not in merged or item["similarity"] > merged[cid]["similarity"]:
                    merged[cid] = {**item, "matched_subquery": part}

        ranked = sorted(
            merged.values(),
            key=lambda r: r["similarity"] * (0.5 + r["coverage"]),
            reverse=True,
        )
        return ranked[:top_k]