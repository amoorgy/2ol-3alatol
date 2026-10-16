"""Mine attested spelling-variant clusters from a Franco corpus.

Words that share a phonemic skeleton are candidate variants of each other;
edit distance to the most frequent form confirms the cluster. The output is
the empirical grounding for the taxonomy weights: rules that never appear in
mined clusters are suspect, frequent cluster patterns deserve higher weight.
"""

from collections import Counter, defaultdict
from typing import Iterable

from .skeleton import skeleton, words


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def mine_variants(
    texts: Iterable[str],
    min_count: int = 2,
    min_word_len: int = 3,
) -> list[dict]:
    """Clusters of attested variant forms, most frequent first.

    Each cluster: {"skeleton": str, "forms": [(form, count), ...], "total": int}
    with forms sorted by descending count.
    """
    counts: Counter[str] = Counter(w for t in texts for w in words(t))
    groups: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for word, count in counts.items():
        if count >= min_count and len(word) >= min_word_len:
            groups[skeleton(word)].append((word, count))

    clusters = []
    for key, forms in groups.items():
        # Single-consonant skeletons over-merge unrelated function words
        # ("ana"/"eny" both collapse to "n"); demand at least two segments.
        if len(forms) < 2 or len(key) < 2:
            continue
        forms.sort(key=lambda fc: (-fc[1], fc[0]))
        head = forms[0][0]
        kept = forms[:1] + [
            (w, c)
            for w, c in forms[1:]
            # Absolute edit cap: spelling variants are close; and a
            # proper prefix/suffix relation ("7elw"/"7elwa",
            # "garreb"/"agarreb") signals inflection, not spelling.
            if edit_distance(w, head) <= 2
            and not w.startswith(head)
            and not head.startswith(w)
            and not w.endswith(head)
            and not head.endswith(w)
        ]
        if len(kept) >= 2:
            clusters.append(
                {"skeleton": key, "forms": kept, "total": sum(c for _, c in kept)}
            )
    clusters.sort(key=lambda c: (-c["total"], c["skeleton"]))
    return clusters


def clusters_to_tsv(clusters: list[dict]) -> str:
    lines = ["skeleton\ttotal\tforms"]
    for c in clusters:
        forms = " ".join(f"{w}:{n}" for w, n in c["forms"])
        lines.append(f"{c['skeleton']}\t{c['total']}\t{forms}")
    return "\n".join(lines) + "\n"
