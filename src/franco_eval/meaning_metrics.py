"""Score the in-context word-meaning probe: agreement across renderings.

The design is a 2x2 and the two failure directions are NEVER collapsed into
one score:

  variants of one word SHOULD agree     -> disagreement = under-robust
  a trap (different word) SHOULD differ -> agreement    = over-normalised

The None asymmetry is deliberate and load-bearing. For variants, a non-answer
IS a failure to demonstrate robustness, which is exactly the semantics
`pairwise_agreement` already has (None never agrees). For traps the same rule
would be wrong: a model that refuses the trap would be scored as having
correctly distinguished it. So traps use a directed comparison against the
modal variant meaning with an explicit `n_traps_scored` denominator, and
the rate is None -- never 0.0 -- when that denominator is empty. Absent
data is not a score of zero.

Stdlib only; the agreement functions come from src/franco/metrics.py unchanged.
"""

import collections
from dataclasses import dataclass, field

from franco.metrics import consistency_at_k, pairwise_agreement


@dataclass(frozen=True)
class ItemMeanings:
    """Parsed, normalised meanings for one probe item under one model.

    Every meaning is either a normalised Arabic string or None (non-answer:
    wrong script, too many tokens, empty, or the call failed).
    """

    item_id: str
    word_type: str
    base: str | None
    variants: tuple[str | None, ...]
    # ortho classes per variant, aligned with `variants`
    variant_classes: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    trap: str | None = None
    has_trap: bool = False
    # the base rendering asked a second time -- the generation-noise ceiling
    base_repeat: str | None = None
    has_repeat: bool = False
    gold: frozenset[str] = frozenset()


def _as_int_groups(groups: list[list[str | None]]) -> list[list[int | None]]:
    """Map meaning strings to ints so franco.metrics keeps its declared types."""
    out: list[list[int | None]] = []
    for group in groups:
        ids: dict[str, int] = {}
        out.append([None if g is None else ids.setdefault(g, len(ids))
                    for g in group])
    return out


def modal_meaning(item: ItemMeanings) -> tuple[str | None, int]:
    """Most common non-None meaning among base+variants, with its count."""
    counts = collections.Counter(
        g for g in (item.base, *item.variants) if g is not None)
    if not counts:
        return None, 0
    meaning, count = counts.most_common(1)[0]
    return meaning, count


def score(items: list[ItemMeanings]) -> dict:
    """All metrics for one (model, arm)."""
    groups = [[item.base, *item.variants] for item in items]
    int_groups = _as_int_groups(groups)

    n_calls = sum(len(g) for g in groups)
    n_calls += sum(1 for item in items if item.has_trap)
    n_calls += sum(1 for item in items if item.has_repeat)
    n_parsed = sum(1 for g in groups for x in g if x is not None)
    n_parsed += sum(1 for item in items if item.has_trap and item.trap is not None)
    n_parsed += sum(1 for item in items
                    if item.has_repeat and item.base_repeat is not None)

    agreement = pairwise_agreement(int_groups) if int_groups else None
    consistency = consistency_at_k(int_groups) if int_groups else None

    modal_share_values = []
    for item in items:
        _meaning, count = modal_meaning(item)
        k = 1 + len(item.variants)
        modal_share_values.append(count / k)

    # --- traps: directed, explicit denominator ------------------------------
    n_traps_scored = 0
    n_over = 0
    n_trap_items = sum(1 for item in items if item.has_trap)
    n_trap_none = sum(1 for item in items if item.has_trap and item.trap is None)
    for item in items:
        if not item.has_trap or item.trap is None:
            continue
        meaning, count = modal_meaning(item)
        if meaning is None:
            continue
        n_traps_scored += 1
        if item.trap == meaning:
            n_over += 1

    # --- ceiling: same rendering asked twice --------------------------------
    n_repeat_scored = 0
    n_repeat_agree = 0
    for item in items:
        if not item.has_repeat:
            continue
        if item.base is None and item.base_repeat is None:
            continue  # double non-answer says nothing about stability
        n_repeat_scored += 1
        if item.base is not None and item.base == item.base_repeat:
            n_repeat_agree += 1

    # --- per-phenomenon flips ------------------------------------------------
    by_class: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"n_edges": 0, "n_flips": 0, "n_none": 0})
    for item in items:
        for meaning, classes in zip(item.variants, item.variant_classes):
            for cls in classes:
                cell = by_class[cls]
                cell["n_edges"] += 1
                if meaning is None or item.base is None:
                    cell["n_none"] += 1
                if meaning is None or item.base is None or meaning != item.base:
                    cell["n_flips"] += 1

    # --- gold anchor, where it exists ---------------------------------------
    n_gold = 0
    n_gold_correct = 0
    for item in items:
        if not item.gold:
            continue
        for meaning in (item.base, *item.variants):
            if meaning is None:
                continue
            n_gold += 1
            if meaning in item.gold:
                n_gold_correct += 1

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "n_items": len(items),
        "n_calls": n_calls,
        "n_parsed": n_parsed,
        "non_answer_rate": rate(n_calls - n_parsed, n_calls),
        "variant_pairwise_agreement": (round(agreement, 4)
                                       if agreement is not None else None),
        "variant_consistency_at_k": (round(consistency, 4)
                                     if consistency is not None else None),
        "under_robust_rate": (round(1.0 - agreement, 4)
                              if agreement is not None else None),
        "modal_meaning_share": (round(sum(modal_share_values)
                                    / len(modal_share_values), 4)
                              if modal_share_values else None),
        "n_trap_items": n_trap_items,
        "n_traps_scored": n_traps_scored,
        "trap_non_answer_rate": rate(n_trap_none, n_trap_items),
        "over_norm_rate": rate(n_over, n_traps_scored),
        "ceiling_n": n_repeat_scored,
        "ceiling_agreement": rate(n_repeat_agree, n_repeat_scored),
        "n_gold": n_gold,
        "meaning_accuracy": rate(n_gold_correct, n_gold),
        "by_class": {
            cls: {"n_edges": cell["n_edges"], "n_none": cell["n_none"],
                  "flip_rate": rate(cell["n_flips"], cell["n_edges"])}
            for cls, cell in sorted(by_class.items())
        },
    }
