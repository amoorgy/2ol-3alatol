"""E1 figures. The only claim each figure makes is the one in its FIGURES entry."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualization import theme
from visualization.series import (arm_dumbbell_series,
                                  fragmentation_ceiling_series,
                                  phenomenon_matrix, separability_series)


def _base(mode: str, figsize):
    plt.rcParams.update(theme.rc(mode))
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _na(ax, y, mode, x=0.01):
    ax.text(x, y, theme.NULL_TEXT, va="center", fontsize=8,
            color=theme.palette(mode)["muted"])


def fig_fragmentation_ceiling(level1: dict, *, arm: str = "human",
                              mode: str = "light"):
    rows = fragmentation_ceiling_series(level1, arm)
    fig, ax = _base(mode, (7, 0.42 * len(rows) + 1.6))
    colors = theme.palette(mode)
    accent = theme.provenance_color(arm, mode)
    for i, row in enumerate(reversed(rows)):
        same, diff = row.values
        if same is not None and diff is not None:
            ax.plot([same, diff], [i, i], color=colors["baseline"], lw=1.2, zorder=1)
        if same is not None:
            ax.plot(same, i, "o", mfc="none", mec=accent, ms=7, zorder=2)
        if diff is not None:
            ax.plot(diff, i, "o", color=accent, ms=7, zorder=2)
        else:
            _na(ax, i, mode, x=(same or 0) + 0.03)
    labels = [f"{r.label} ({r.note})" if r.group == "control" else r.label
              for r in reversed(rows)]
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean fragmentation (1 − token-set Jaccard)")
    ax.set_title(f"Fragmentation on SAME (○) vs DIFFERENT (●) pairs — {arm} arm",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_separability_auc(level1: dict, *, arm: str = "human", mode: str = "light"):
    rows = separability_series(level1, arm)
    fig, ax = _base(mode, (7, 0.42 * len(rows) + 1.6))
    accent = theme.provenance_color(arm, mode)
    for i, row in enumerate(rows):
        auc, lo, hi = row.values
        if auc is None:
            _na(ax, i, mode, x=0.505)
            continue
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], color=accent, lw=3, alpha=0.35, zorder=1)
        ax.plot(auc, i, "o", color=accent, ms=7, zorder=2)
    ax.axvline(0.5, color=theme.palette(mode)["ink_2"], lw=1.4, ls="--")
    ax.text(0.5, len(rows) - 0.3, " chance", fontsize=8,
            color=theme.palette(mode)["ink_2"])
    ax.set_yticks(range(len(rows)), [r.label for r in rows])
    ax.set_xlabel("separability AUC (DIFFERENT fragments more than SAME)")
    ax.set_title(f"Can fragmentation separate variants from look-alikes? — {arm} arm",
                 fontsize=10)
    fig.tight_layout()
    return fig


def fig_phenomenon_heatmap(level1: dict, *, arm: str = "nilechat",
                           mode: str = "light"):
    phenomena, tokenizers, values, ns = phenomenon_matrix(level1, arm)
    fig, ax = _base(mode, (1.1 * len(tokenizers) + 2.4, 0.5 * len(phenomena) + 1.8))
    colors = theme.palette(mode)
    ramp = theme.SEQ_BLUE
    for r, phenomenon in enumerate(phenomena):
        for c, _tok in enumerate(tokenizers):
            value, n = values[r][c], ns[r][c]
            if value is None:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False,
                                           hatch=theme.HATCH_SMALL_N,
                                           edgecolor=colors["grid"]))
                ax.text(c + 0.5, r + 0.5, theme.NULL_TEXT, ha="center",
                        va="center", fontsize=7, color=colors["muted"])
                continue
            shade = ramp[min(int(value * len(ramp)), len(ramp) - 1)]
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=shade,
                                       alpha=theme.cell_alpha(n)))
            ink = "#ffffff" if value > 0.55 else colors["ink"]
            note = theme.annotate_n(n)
            ax.text(c + 0.5, r + 0.45, f"{value:.2f}", ha="center", va="center",
                    fontsize=8, color=ink)
            if note:
                ax.text(c + 0.5, r + 0.78, note, ha="center", va="center",
                        fontsize=6, color=ink)
    ax.set_xlim(0, len(tokenizers))
    ax.set_ylim(len(phenomena), 0)
    ax.set_xticks([c + 0.5 for c in range(len(tokenizers))], tokenizers,
                  rotation=30, ha="right", fontsize=8)
    ax.set_yticks([r + 0.5 for r in range(len(phenomena))], phenomena, fontsize=8)
    ax.grid(False)
    ax.set_title(f"Mean fragmentation per phenomenon (SAME edges) — {arm} arm")
    fig.text(0.01, 0.01, theme.SMALL_N_FOOTNOTE, fontsize=7,
             color=colors["muted"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_arm_dumbbell(level1: dict, *, mode: str = "light"):
    rows = arm_dumbbell_series(level1)
    fig, ax = _base(mode, (7, 0.42 * len(rows) + 1.6))
    human = theme.provenance_color("human", mode)
    nile = theme.provenance_color("nilechat", mode)
    for i, row in enumerate(reversed(rows)):
        h, n = row.values
        if h is not None and n is not None:
            ax.plot([h, n], [i, i], color=theme.palette(mode)["baseline"],
                    lw=1.2, zorder=1)
        if h is not None:
            ax.plot(h, i, "o", color=human, ms=7, zorder=2)
        if n is not None:
            ax.plot(n, i, "o", color=nile, ms=7, zorder=2)
    ax.set_yticks(range(len(rows)), [r.label for r in reversed(rows)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean fragmentation on SAME pairs")
    ax.set_title("Does the headline hold on the human-authored arm?", fontsize=10)
    ax.plot([], [], "o", color=human, label="human (authored)")
    ax.plot([], [], "o", color=nile, label="nilechat (machine-derived)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
