"""Do tokenizers see two spellings of one word as the same thing?

Stdlib only. Every function takes an `encode` callable mapping text to token
ids, so the metrics are testable against a fake tokenizer and are not coupled
to transformers.

The central quantity is *fragmentation*: how little two spellings of one word
share once tokenized. A fragmentation of 1.0 means the model receives no
overlapping evidence at all that `a5sam` and `akhsam` are the same word.
"""

import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

Encode = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class EdgeFragmentation:
    """One spelling pair as seen by one tokenizer."""

    a: str
    b: str
    ids_a: tuple[int, ...]
    ids_b: tuple[int, ...]
    classes: tuple[str, ...] = ()

    @property
    def same_seq(self) -> bool:
        """Did both spellings produce an identical token sequence?"""
        return self.ids_a == self.ids_b

    @property
    def len_delta(self) -> int:
        return abs(len(self.ids_a) - len(self.ids_b))

    @property
    def jaccard(self) -> float:
        """Shared token vocabulary between the two spellings."""
        sa, sb = set(self.ids_a), set(self.ids_b)
        union = sa | sb
        if not union:
            return 1.0
        return len(sa & sb) / len(union)

    @property
    def frag(self) -> float:
        """1.0 = the two spellings share no tokens; 0.0 = identical vocabulary."""
        return 1.0 - self.jaccard


def fragment_edge(
    encode: Encode, a: str, b: str, classes: Sequence[str] = ()
) -> EdgeFragmentation:
    return EdgeFragmentation(a, b, tuple(encode(a)), tuple(encode(b)), tuple(classes))


def fertility(encode: Encode, words: Iterable[str]) -> float:
    """Mean tokens per word. Higher means the tokenizer shreds this script."""
    counts = [len(encode(w)) for w in words if w]
    return statistics.fmean(counts) if counts else 0.0


@dataclass(frozen=True)
class ClassSummary:
    phenomenon: str
    n_edges: int
    mean_frag: float
    same_seq_rate: float
    mean_len_delta: float


def aggregate_by_class(edges: Iterable[EdgeFragmentation]) -> list[ClassSummary]:
    """Group edges by phenomenon. An edge with two classes counts under both."""
    buckets: dict[str, list[EdgeFragmentation]] = {}
    for edge in edges:
        for label in edge.classes or ("UNLABELLED",):
            buckets.setdefault(label, []).append(edge)
    out = [
        ClassSummary(
            phenomenon=label,
            n_edges=len(rows),
            mean_frag=statistics.fmean(r.frag for r in rows),
            same_seq_rate=statistics.fmean(1.0 if r.same_seq else 0.0 for r in rows),
            mean_len_delta=statistics.fmean(float(r.len_delta) for r in rows),
        )
        for label, rows in buckets.items()
    ]
    return sorted(out, key=lambda s: (-s.mean_frag, s.phenomenon))


def auc(higher: Sequence[float], lower: Sequence[float]) -> float:
    """Probability that a value drawn from `higher` exceeds one from `lower`.

    Rank-based (Mann-Whitney) with ties credited half. Used to ask whether
    fragmentation can separate the DIFFERENT pairs from the SAME ones: 0.5 means
    the score carries no information, so a tokenizer offers the model no help in
    telling a real spelling variant from a look-alike different word.
    """
    if not higher or not lower:
        return 0.5
    wins = 0.0
    for h in higher:
        for l in lower:
            wins += 1.0 if h > l else (0.5 if h == l else 0.0)
    return wins / (len(higher) * len(lower))
