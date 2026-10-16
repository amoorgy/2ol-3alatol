"""Distributional evidence for whether two spellings are the same word.

`classify_edge` answers "can these two spellings be related by orthographic
rules?", which in Arabizi is nearly always yes: vowels are optional, so almost
any two short words sharing a consonant frame converge. Measured on NileChat,
93% of mined co-skeleton pairs converge, and that bucket mixes true variants
(`men`/`min`) with different words (`men`/`meen`, من / مين) indistinguishably.

Words that mean the same thing appear in the same company, so context supplies
the signal spelling cannot. Position-tagged PPMI vectors separate the two
classes at AUC ~0.93 on a hand-checked probe -- good enough to RANK candidates
for review, not good enough to LABEL them (`2ol`/`2ool` scores 0.154, a false
alarm). Nothing here decides that two words differ; it decides what a reviewer
should look at first.

The threshold is calibrated on data/benchmark/pairs.csv -- the human-authored
arm -- which is what hard rule 1 asks for.

Stdlib only.
"""

import collections
import csv
import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from franco_eval.tokenizer_metrics import auc

Feature = tuple[int, int]  # (signed offset, context word id)


@dataclass(frozen=True)
class ContextModel:
    """PPMI-weighted context vectors, L2 norms precomputed for cosine."""

    vectors: dict[str, dict[Feature, float]]
    norms: dict[str, float]
    n_contexts: int

    def __contains__(self, word: str) -> bool:
        return self.norms.get(word, 0.0) > 0.0

    def cosine(self, a: str, b: str) -> float | None:
        """None when either word lacks a usable vector -- never a silent 0.0."""
        va, vb = self.vectors.get(a), self.vectors.get(b)
        na, nb = self.norms.get(a, 0.0), self.norms.get(b, 0.0)
        if not va or not vb or na == 0.0 or nb == 0.0:
            return None
        if len(va) > len(vb):
            va, vb = vb, va
        dot = sum(weight * vb.get(feature, 0.0) for feature, weight in va.items())
        return dot / (na * nb)


def context_vocabulary(counts: dict[str, int], skip_top: int = 20,
                       size: int = 1500) -> dict[str, int]:
    """Context words by frequency rank, dropping the most common ones.

    `el`, `w` and `fe` occur beside everything, so they carry no information
    about which word sits in the slot while dominating every vector. Skipping
    the top 20 is what makes the separation work at all.
    """
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {word: index for index, (word, _n) in enumerate(ranked[skip_top:skip_top + size])}


def accumulate(sentence_words: Iterable[Sequence[str]], targets: set[str],
               ctx_index: dict[str, int], window: int = 2):
    """Co-occurrence counts for the targets over tokenised sentences.

    Offsets are kept signed and separate: `el kitab` and `kitab el` are
    different evidence, and collapsing them loses the word-order cue that
    separates a noun from the verb spelled almost the same way.
    """
    pair_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    feature_totals: collections.Counter = collections.Counter()
    target_totals: collections.Counter = collections.Counter()
    total = 0
    for words in sentence_words:
        for position, word in enumerate(words):
            if word not in targets:
                continue
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbour_at = position + offset
                if not 0 <= neighbour_at < len(words):
                    continue
                ctx_id = ctx_index.get(words[neighbour_at])
                if ctx_id is None:
                    continue
                feature = (offset, ctx_id)
                pair_counts[word][feature] += 1
                feature_totals[feature] += 1
                target_totals[word] += 1
                total += 1
    return pair_counts, feature_totals, target_totals, total


def build_model(pair_counts, feature_totals, target_totals, total,
                min_count: int = 2) -> ContextModel:
    """Positive PMI, then L2 norms.

    PPMI rather than raw counts because a shared rare neighbour is far stronger
    evidence of shared meaning than a shared common one, and negative PMI on
    sparse counts is mostly noise.
    """
    vectors: dict[str, dict[Feature, float]] = {}
    norms: dict[str, float] = {}
    for word, counts in pair_counts.items():
        word_total = target_totals[word]
        if word_total < min_count:
            continue
        vector: dict[Feature, float] = {}
        for feature, joint in counts.items():
            if joint < min_count:
                continue
            pmi = math.log((joint * total) / (word_total * feature_totals[feature]))
            if pmi > 0.0:
                vector[feature] = pmi
        if vector:
            vectors[word] = vector
            norms[word] = math.sqrt(sum(w * w for w in vector.values()))
    return ContextModel(vectors, norms, len(feature_totals))


# --- calibration on the human-authored arm ----------------------------------

@dataclass(frozen=True)
class Calibration:
    threshold: float
    auc: float
    n_same: int
    n_different: int
    n_skipped: int

    @property
    def usable(self) -> bool:
        """Below ~0.7 the ranking is not worth a reviewer's attention."""
        return self.auc >= 0.7 and self.n_same >= 10 and self.n_different >= 10


def calibrate(model: ContextModel, pairs_csv: pathlib.Path) -> Calibration:
    """Threshold and AUC from the human-authored pairs, not from NileChat.

    Uses only rows where both spellings have vectors; `n_skipped` records the
    rest so coverage is visible rather than assumed.
    """
    same: list[float] = []
    different: list[float] = []
    skipped = 0
    with pairs_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = model.cosine(row["a"].lower(), row["b"].lower())
            if score is None:
                skipped += 1
                continue
            (same if row["label"] == "SAME" else different).append(score)

    if not same or not different:
        return Calibration(1.0, 0.5, len(same), len(different), skipped)

    # auc(higher, lower): probability a SAME pair scores above a DIFFERENT one.
    score = auc(same, different)
    best = (-1.0, 1.0)
    for candidate in sorted(set(same + different)):
        hit = sum(1 for value in same if value >= candidate) / len(same)
        fa = sum(1 for value in different if value >= candidate) / len(different)
        youden = hit - fa
        if youden > best[0]:
            best = (youden, candidate)
    return Calibration(round(best[1], 4), round(score, 4), len(same), len(different), skipped)
