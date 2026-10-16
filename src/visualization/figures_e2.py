"""E2 (level 3) figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualization import theme
from visualization.series import (agreement_ladder_series, capability_tiles,
                                  flip_by_phenomenon_panels, robustness_points)

LADDER_COLUMNS = ("pairwise agreement", "consistency@k (all identical)",
                  "same-rendering ceiling", "answered rate")


def fig_robustness_scatter(level3: dict, *, mode: str = "light"):
    """THE headline. Two failure axes, structurally impossible to collapse."""
    points = robustness_points(level3)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    plt.rcParams.update(theme.rc(mode))
    colors = theme.palette(mode)
    accent = theme.provenance_color("nilechat", mode)
    for point in points:
        if point.x is None:
            continue
        if point.y is None:
            # No scorable traps: the point lives in a labelled gutter, never at 0.
            ax.plot(point.x, -0.08, "s", color=colors["muted"], ms=8, clip_on=False)
            ax.annotate(f"{point.label} (traps n/a)", (point.x, -0.08),
                        textcoords="offset points", xytext=(6, -2), fontsize=8,
                        color=colors["muted"], annotation_clip=False)
            continue
        ax.plot(point.x, point.y, "o", color=accent, ms=9)
        ax.annotate(point.label, (point.x, point.y),
                    textcoords="offset points", xytext=(7, 4), fontsize=8,
                    color=colors["ink"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("under-robust rate (variants disagree — brittle)")
    ax.set_ylabel("over-normalisation rate (trap read as the same word)")
    ax.text(0.02, 0.02, "ideal", fontsize=9, color=colors["ink_2"])
    ax.text(0.72, 0.95, "collapses everything", fontsize=8, color=colors["muted"])
    ax.text(0.72, 0.02, "distinguishes but brittle", fontsize=8,
            color=colors["muted"])
    ax.set_title("Two separate failures — never one score")
    fig.tight_layout()
    return fig


def fig_agreement_ladder(level3: dict, *, mode: str = "light"):
    rows = agreement_ladder_series(level3)
    fig, ax = plt.subplots(figsize=(7.4, 0.9 * len(rows) + 1.8))
    plt.rcParams.update(theme.rc(mode))
    colors = theme.palette(mode)
    accent = theme.provenance_color("nilechat", mode)
    bar_height = 0.18
    for i, row in enumerate(rows):
        for j, value in enumerate(row.values):
            y = i + (j - 1.5) * bar_height
            if value is None:
                ax.text(0.01, y, theme.NULL_TEXT, va="center", fontsize=7,
                        color=colors["muted"])
                continue
            ax.barh(y, value, height=bar_height * 0.88, color=accent,
                    alpha=1.0 - 0.2 * j)
            ax.text(value + 0.01, y, f"{value:.2f}", va="center", fontsize=7,
                    color=colors["ink_2"])
    ax.set_yticks(range(len(rows)), [r.label for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    handles = [plt.Rectangle((0, 0), 1, 1, color=accent, alpha=1.0 - 0.2 * j)
               for j in range(4)]
    ax.legend(handles, LADDER_COLUMNS, loc="lower right", frameon=False,
              fontsize=7)
    ax.set_title("Why pairwise agreement is the headline (and C@k the floor note)")
    fig.tight_layout()
    return fig


def fig_flip_by_phenomenon(level3: dict, *, mode: str = "light"):
    panels = flip_by_phenomenon_panels(level3)
    if not panels:
        return None
    n = len(panels)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.1 * nrows),
                             sharex=True, squeeze=False)
    plt.rcParams.update(theme.rc(mode))
    colors = theme.palette(mode)
    accent = theme.provenance_color("nilechat", mode)
    flat = [ax for row in axes for ax in row]
    for ax, (model, rows) in zip(flat, sorted(panels.items())):
        for i, row in enumerate(rows):
            value = row.values[0]
            if value is None:
                ax.text(0.01, i, theme.NULL_TEXT, va="center", fontsize=7,
                        color=colors["muted"])
                continue
            ax.barh(i, value, color=accent, alpha=theme.cell_alpha(row.n[0]))
            note = theme.annotate_n(row.n[0])
            if note:
                ax.text(value + 0.01, i, note, va="center", fontsize=6,
                        color=colors["muted"])
        ax.set_yticks(range(len(rows)), [r.label for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(model, fontsize=9)
    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Spelling-induced meaning flips, per phenomenon")
    fig.text(0.01, 0.01, theme.SMALL_N_FOOTNOTE, fontsize=7, color=colors["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def fig_capability_floor(level3: dict, *, mode: str = "light"):
    """Stat tiles, not a chart: one hero number + verdict per model."""
    tiles = capability_tiles(level3)
    fig, axes = plt.subplots(1, len(tiles), figsize=(2.4 * len(tiles), 2.3))
    plt.rcParams.update(theme.rc(mode))
    colors = theme.palette(mode)
    if len(tiles) == 1:
        axes = [axes]
    for ax, (model, accuracy, passed) in zip(axes, tiles):
        ax.axis("off")
        value = theme.NULL_TEXT if accuracy is None else f"{accuracy:.0%}"
        verdict = "PASS" if passed else "FAIL"
        verdict_color = (theme.provenance_color("human", mode) if passed
                         else theme.provenance_color("nilechat", mode))
        ax.text(0.5, 0.72, value, ha="center", fontsize=22, color=colors["ink"])
        ax.text(0.5, 0.42, verdict, ha="center", fontsize=10,
                color=verdict_color, fontweight="bold")
        ax.text(0.5, 0.16, model, ha="center", fontsize=8, color=colors["ink_2"])
    fig.suptitle("Arabic-script capability floor (30 words): orthography claims"
                 " need a capable model", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
