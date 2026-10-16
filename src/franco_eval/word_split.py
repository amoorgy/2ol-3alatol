"""Held-out word split for the v2 training set: learn the rule, not the pairs.

The v2 fine-tune is meant to learn two rules at once -- licensed respellings
of one word mean the same thing, and look-alike trap words do not. The only
way to tell a learned rule from memorised pairs is to measure on words the
model never trained on, so the benchmark words are split in two and the
held-out half (with every trap attached to it, plus the whole human arm) is
firewalled out of both datasets.

A benchmark word and its trap are one unit: a pair straddling the split
would leak the held-out side through the train side's contrast sentences.
Words are grouped into connected components over the trap pairs (one trap
can pair with several words, and some traps are benchmark words
themselves), and each component lands on one side by a salted hash of its
smallest member -- deterministic, no RNG state, independent of input order.

Stdlib only.
"""

import hashlib
from typing import Iterable

TRAIN = "train"
HELDOUT = "heldout"


def components(words: Iterable[str],
               pairs: Iterable[tuple[str, str]]) -> list[frozenset[str]]:
    """Connected components of `words` joined by `pairs` (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for word in words:
        find(word)
    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    groups: dict[str, set[str]] = {}
    for word in parent:
        groups.setdefault(find(word), set()).add(word)
    return sorted((frozenset(g) for g in groups.values()), key=min)


def _hash_unit(salt: str, key: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8]
    return int(digest, 16) / 16 ** 8


def split_words(words: Iterable[str], pairs: Iterable[tuple[str, str]], *,
                salt: str, heldout_fraction: float = 0.5,
                forced_heldout: frozenset[str] = frozenset()) -> dict[str, str]:
    """word -> TRAIN | HELDOUT, one side per component.

    `forced_heldout` words (the human arm) are always held out, and so is
    any component that touches one of them.
    """
    sides: dict[str, str] = {}
    for comp in components([*words, *forced_heldout], pairs):
        if comp & forced_heldout:
            side = HELDOUT
        else:
            side = (HELDOUT if _hash_unit(salt, min(comp)) < heldout_fraction
                    else TRAIN)
        for word in comp:
            sides[word] = side
    return sides
