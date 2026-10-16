"""Results JSON -> figure-ready series. STDLIB ONLY, fully unit-testable.

Two rules every function here obeys:

  - None is a legitimate metric value. It is never turned into 0, never
    dropped silently, never interpolated. It flows through as None and the
    figure layer draws an explicit "n/a".
  - Every value travels with its n, so the figure layer can fade small cells.
"""

import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class DotRow:
    """One row of a dot/bar chart: a label and one value per column."""

    label: str
    values: tuple[float | None, ...]
    n: tuple[int, ...] = ()
    group: str = ""
    note: str = ""


@dataclass(frozen=True)
class ScatterPoint:
    x: float | None
    y: float | None
    label: str
    arm: str = ""
    family: str = ""
    n: int = 0


# --- statistics ---------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float],
                 *, boot: int = 2000, seed: int = 13) -> tuple[float, float] | None:
    """Percentile bootstrap; deterministic under the project seed."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    scores = sorted(stat([rng.choice(values) for _ in values])
                    for _ in range(boot))
    return (round(scores[int(0.025 * boot)], 4),
            round(scores[int(0.975 * boot)], 4))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation via stdlib; None (not a crash) on degenerate input."""
    if len(x) != len(y) or len(x) < 3:
        return None
    try:
        return round(statistics.correlation(list(x), list(y), method="ranked"), 4)
    except statistics.StatisticsError:
        return None


# --- E1 series -----------------------------------------------------------------

def fragmentation_ceiling_series(level1: dict, arm: str) -> list[DotRow]:
    """(frag SAME, frag DIFFERENT) per tokenizer; controls carry group."""
    rows = []
    for name, entry in level1["tokenizers"].items():
        r = entry["arms"].get(arm)
        if r is None:
            continue
        rows.append(DotRow(
            label=name,
            values=(r["mean_frag_same"], r["mean_frag_different"]),
            n=(r["n_same"], r["n_different"]),
            group=entry["group"],
            note=entry["family"],
        ))
    rows.sort(key=lambda row: (row.group != "production",
                               -(row.values[0] or 0.0)))
    return rows


def separability_series(level1: dict, arm: str) -> list[DotRow]:
    """(auc, ci_lo, ci_hi) per tokenizer."""
    rows = []
    for name, entry in level1["tokenizers"].items():
        r = entry["arms"].get(arm)
        if r is None:
            continue
        ci = r.get("separability_ci95") or (None, None)
        rows.append(DotRow(label=name,
                           values=(r["separability_auc"], ci[0], ci[1]),
                           n=(r["n_edges"],), group=entry["group"],
                           note=entry["family"]))
    rows.sort(key=lambda row: row.values[0] if row.values[0] is not None else 0.5)
    return rows


def phenomenon_matrix(level1: dict, arm: str) -> tuple[list[str], list[str],
                                                       list[list[float | None]],
                                                       list[list[int]]]:
    """(phenomena, tokenizers, mean_frag matrix, n matrix), rows sorted by
    mean fragmentation descending so the worst class reads first."""
    tokenizers = [name for name, entry in level1["tokenizers"].items()
                  if arm in entry["arms"]]
    classes: dict[str, dict[str, tuple[float, int]]] = {}
    for name in tokenizers:
        for cell in level1["tokenizers"][name]["arms"][arm]["by_class"]:
            classes.setdefault(cell["phenomenon"], {})[name] = (
                cell["mean_frag"], cell["n_edges"])
    def row_mean(phenomenon: str) -> float:
        cells = [v[0] for v in classes[phenomenon].values() if v[0] is not None]
        return statistics.fmean(cells) if cells else 0.0
    phenomena = sorted(classes, key=row_mean, reverse=True)
    values = [[classes[p].get(t, (None, 0))[0] for t in tokenizers]
              for p in phenomena]
    ns = [[classes[p].get(t, (None, 0))[1] for t in tokenizers]
          for p in phenomena]
    return phenomena, tokenizers, values, ns


def arm_dumbbell_series(level1: dict) -> list[DotRow]:
    """(human frag SAME, nilechat frag SAME) per tokenizer -- the rule-1 figure."""
    rows = []
    for name, entry in level1["tokenizers"].items():
        human = entry["arms"].get("human")
        nile = entry["arms"].get("nilechat")
        if human is None and nile is None:
            continue
        rows.append(DotRow(
            label=name,
            values=(human["mean_frag_same"] if human else None,
                    nile["mean_frag_same"] if nile else None),
            n=(human["n_same"] if human else 0, nile["n_same"] if nile else 0),
            group=entry["group"], note=entry["family"]))
    rows.sort(key=lambda row: -(row.values[0] or 0.0))
    return rows


# --- E2 series -----------------------------------------------------------------

def robustness_points(level3: dict) -> list[ScatterPoint]:
    """The headline scatter: x=under-robust, y=over-norm, one point per model.

    A model with over_norm_rate None (no scorable traps) yields y=None; the
    figure layer places it in a side gutter rather than pretending it is 0.
    """
    points = []
    for model, r in level3["models"].items():
        if "n_items" not in r:
            continue
        points.append(ScatterPoint(
            x=r.get("under_robust_rate"), y=r.get("over_norm_rate"),
            label=model, arm="nilechat", n=r.get("n_items", 0)))
    return points


def agreement_ladder_series(level3: dict) -> list[DotRow]:
    """Why pairwise agreement is the headline and C@k the ceiling note."""
    rows = []
    for model, r in level3["models"].items():
        if "n_items" not in r:
            continue
        rows.append(DotRow(
            label=model,
            values=(r.get("variant_pairwise_agreement"),
                    r.get("variant_consistency_at_k"),
                    r.get("ceiling_agreement"),
                    None if r.get("non_answer_rate") is None
                    else round(1.0 - r["non_answer_rate"], 4)),
            n=(r.get("n_items", 0),) * 4))
    rows.sort(key=lambda row: -(row.values[0] or 0.0))
    return rows


def flip_by_phenomenon_panels(level3: dict) -> dict[str, list[DotRow]]:
    """model -> per-class flip rates, shared class order across panels."""
    order: dict[str, float] = {}
    for r in level3["models"].values():
        for cls, cell in r.get("by_class", {}).items():
            if cell["flip_rate"] is not None:
                order[cls] = max(order.get(cls, 0.0), cell["flip_rate"])
    classes = sorted(order, key=order.get, reverse=True)
    panels = {}
    for model, r in level3["models"].items():
        if "by_class" not in r:
            continue
        panels[model] = [
            DotRow(label=cls,
                   values=(r["by_class"].get(cls, {}).get("flip_rate"),),
                   n=(r["by_class"].get(cls, {}).get("n_edges", 0),))
            for cls in classes]
    return panels


def capability_tiles(level3: dict) -> list[tuple[str, float | None, bool]]:
    """(model, floor accuracy, passed) -- a stat-tile row, not a chart."""
    tiles = []
    for model, r in level3["models"].items():
        control = r.get("control")
        if control is None:
            tiles.append((model, None, False))
        else:
            tiles.append((model, control["accuracy"],
                          control["capability"] == "PASS"))
    return tiles


# --- joint ----------------------------------------------------------------------

def frag_vs_flip_points(level1: dict, level3: dict, *,
                        model_to_tokenizer: dict[str, str],
                        arm: str = "nilechat") -> tuple[list[ScatterPoint],
                                                        float | None]:
    """One point per (model x phenomenon): x = that model's own tokenizer's
    mean fragmentation for the class, y = the model's flip rate for it.

    The pre-registered E4 question. Drawn whatever the answer is -- committing
    to publish a null is what makes the pre-registration real.
    """
    points = []
    for model, r in level3["models"].items():
        tokenizer = model_to_tokenizer.get(model)
        entry = level1["tokenizers"].get(tokenizer) if tokenizer else None
        if entry is None or arm not in entry["arms"] or "by_class" not in r:
            continue
        frag_by_class = {cell["phenomenon"]: (cell["mean_frag"], cell["n_edges"])
                         for cell in entry["arms"][arm]["by_class"]}
        for cls, cell in r["by_class"].items():
            frag = frag_by_class.get(cls)
            if frag is None or cell["flip_rate"] is None:
                continue
            points.append(ScatterPoint(x=frag[0], y=cell["flip_rate"],
                                       label=f"{model}:{cls}", arm=arm,
                                       family=cls, n=cell["n_edges"]))
    xs = [p.x for p in points if p.x is not None and p.y is not None]
    ys = [p.y for p in points if p.x is not None and p.y is not None]
    return points, spearman(xs, ys)
