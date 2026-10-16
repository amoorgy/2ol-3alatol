"""Farthest-point selection among candidate renderings of one sentence.

Dataset A generates several spelling permutations per sentence via
respell.sample_rendering(). Plain rejection-sampling (dedup exact-string
duplicates only) lets near-identical permutations through -- renderings that
differ from the base by a single word teach a fine-tuned model to parrot the
carrier sentence rather than generalize across spellings.

Renderings are compared by their `choices` dict (word -> chosen rendering),
not by re-tokenizing rendered text: every candidate for one sentence comes
from sample_rendering() over the same spans, so choices dicts share the same
key space by construction, and a missing key means "kept base spelling".

Stdlib only.
"""

from typing import Mapping, Sequence


def pairwise_distance(a: Mapping[str, str], b: Mapping[str, str],
                       words: Sequence[str]) -> int:
    """Count of words where two renderings' choices disagree.

    `words` should be the deduplicated set of respellable words for the
    sentence both renderings came from (e.g. the distinct span words) --
    passing a word twice double-counts it.
    """
    return sum(1 for w in words if a.get(w, w) != b.get(w, w))


def select_diverse(anchors: Sequence[Mapping[str, str]],
                    candidates: Sequence[Mapping[str, str]],
                    words: Sequence[str], k: int, *,
                    min_distance_from_base: int = 2) -> list[int]:
    """Greedy farthest-point (maximin) selection of up to k candidate indices.

    `anchors` are already-fixed renderings a new pick must stay diverse
    from but which are never themselves returned -- typically the base
    sentence (`{}`) alone, or the base plus any renderings already kept
    from an earlier generation pass (so a re-run doesn't add something
    redundant with what's already accepted). Candidates within
    `min_distance_from_base` words of every anchor are dropped outright as
    near-copies -- never selected even to fill out a short quota. Among the
    rest, each step adds whichever remaining candidate maximizes its
    minimum distance to every anchor and everything already picked, the
    standard maximin heuristic for a diverse subset.

    Deterministic: ties break on candidate index, so the same candidate
    list in the same order always yields the same selection.
    """
    def distance_to_anchors(c: Mapping[str, str]) -> int:
        return min(pairwise_distance(a, c, words) for a in anchors)

    eligible = [i for i, c in enumerate(candidates)
                if distance_to_anchors(c) >= min_distance_from_base]
    min_dist = {i: distance_to_anchors(candidates[i]) for i in eligible}
    picked: list[int] = []
    while eligible and len(picked) < k:
        best = max(eligible, key=lambda i: (min_dist[i], -i))
        picked.append(best)
        eligible.remove(best)
        for i in eligible:
            d = pairwise_distance(candidates[best], candidates[i], words)
            if d < min_dist[i]:
                min_dist[i] = d
    return picked
