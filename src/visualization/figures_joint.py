"""The pre-registered E4 question: does fragmentation explain comprehension?

Drawn whatever the answer is. If rho is ~0, the null IS the finding and no
further slicing happens -- the commitment is in the design spec.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualization import theme
from visualization.series import frag_vs_flip_points

# Ollama tag -> E1 registry row. Asserted one-to-one by tests/test_registry.py.
MODEL_TO_TOKENIZER = {
    "aya-expanse:8b": "aya-expanse-8b",
    "gemma4:e2b": "gemma-4-e2b",
    "qwen3.5:9b": "qwen3.5-9b",
    "llama3.1:8b": "llama-3.1-8b",
}


def fig_frag_vs_flip(level1: dict, level3: dict, *, arm: str = "nilechat",
                     mode: str = "light"):
    points, rho = frag_vs_flip_points(level1, level3,
                                      model_to_tokenizer=MODEL_TO_TOKENIZER,
                                      arm=arm)
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 6))
    plt.rcParams.update(theme.rc(mode))
    colors = theme.palette(mode)
    accent = theme.provenance_color(arm, mode)
    for point in points:
        ax.plot(point.x, point.y, "o", color=accent,
                alpha=theme.cell_alpha(point.n), ms=7)
        ax.annotate(point.family, (point.x, point.y),
                    textcoords="offset points", xytext=(5, 3), fontsize=6,
                    color=colors["muted"])
    rho_text = "ρ = n/a" if rho is None else f"Spearman ρ = {rho:.2f}"
    ax.text(0.03, 0.95, rho_text, transform=ax.transAxes, fontsize=11,
            color=colors["ink"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("tokenizer fragmentation for the phenomenon (own tokenizer)")
    ax.set_ylabel("meaning flip rate for the phenomenon")
    ax.set_title("Pre-registered: does fragmentation explain comprehension failure?")
    fig.text(0.01, 0.01, theme.SMALL_N_FOOTNOTE, fontsize=7, color=colors["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
