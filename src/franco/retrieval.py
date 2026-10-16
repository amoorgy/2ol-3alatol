"""Character-n-gram TF-IDF retrieval baseline (pure stdlib).

The easiest serious baseline for noisy short text: index documents as bags
of character trigrams, score queries by cosine similarity. Robustness story:
spelling variants shatter exact trigrams, so raw retrieval degrades with
variation — normalization should recover part of it. Measuring exactly how
much is the showcase experiment.
"""

import math
from collections import Counter, defaultdict


class CharNgramIndex:
    def __init__(self, n: int = 3):
        self.n = n
        self._postings: dict[str, list[tuple[int, float]]] = {}
        self._idf: dict[str, float] = {}
        self._norms: list[float] = []

    def _grams(self, text: str) -> Counter[str]:
        padded = f" {' '.join(text.lower().split())} "
        n = self.n
        if len(padded) < n:
            return Counter([padded])
        return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))

    def fit(self, docs: list[str]) -> "CharNgramIndex":
        doc_grams = [self._grams(d) for d in docs]
        df: Counter[str] = Counter()
        for grams in doc_grams:
            df.update(grams.keys())
        total = len(docs)
        idf = {g: math.log((1 + total) / (1 + c)) + 1.0 for g, c in df.items()}

        postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        norms: list[float] = []
        for doc_id, grams in enumerate(doc_grams):
            weights = {g: c * idf[g] for g, c in grams.items()}
            norms.append(math.sqrt(sum(w * w for w in weights.values())) or 1.0)
            for g, w in weights.items():
                postings[g].append((doc_id, w))
        self._postings = dict(postings)
        self._idf = idf
        self._norms = norms
        return self

    def search(self, query: str, topk: int = 1) -> list[tuple[int, float]]:
        """(doc_id, cosine) pairs, best first; deterministic tie-break on id."""
        if not self._norms:
            raise RuntimeError("CharNgramIndex.search() called before fit()")
        grams = self._grams(query)
        q_weights = {g: c * self._idf.get(g, 0.0) for g, c in grams.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

        scores: defaultdict[int, float] = defaultdict(float)
        for g, qw in q_weights.items():
            for doc_id, dw in self._postings.get(g, ()):
                scores[doc_id] += qw * dw
        ranked = sorted(
            ((doc_id, s / (q_norm * self._norms[doc_id])) for doc_id, s in scores.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:topk]
