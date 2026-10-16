"""Palette, small-n rules and rcParams. STDLIB ONLY -- no matplotlib here.

The palette was validated with a CVD simulator, not eyeballed. Two findings
drive everything:

  - No 4-colour set passes all-pairs distinguishability in dark mode, so
    colour is NEVER the model channel. Models get facets, direct labels and
    marker shapes.
  - Two-colour provenance (human blue vs nilechat orange) passes both modes,
    so the dimension hard rule 1 demands be visible everywhere gets the
    strongest visual channel.
"""

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_2": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7"}
DARK = {"surface": "#1a1a19", "ink": "#ffffff", "ink_2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835"}

# provenance -> (light, dark)
PROVENANCE = {"human": ("#2a78d6", "#3987e5"),
              "nilechat": ("#eb6834", "#d95926")}
GROUP = {"production": ("#2a78d6", "#3987e5"),
         "control": ("#eb6834", "#d95926")}

# 13-step single-hue blue ramp for heatmaps. Never a rainbow.
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#a1c4f0", "#8bb5eb", "#75a6e5",
            "#5f97e0", "#4a88da", "#3a79cf", "#2f6abd", "#265baa",
            "#1d4c97", "#154184", "#0d366b")

MARKERS = {"cohere": "o", "google": "s", "qwen": "^", "meta": "D",
           "dialect": "v", "mistral": "<", "openai": ">",
           "byte-level": "P", "unigram-lm": "X"}

NULL_TEXT = "n/a"
HATCH_SMALL_N = "///"


def palette(mode: str) -> dict[str, str]:
    return LIGHT if mode == "light" else DARK


def provenance_color(arm: str, mode: str) -> str:
    pair = PROVENANCE.get(arm, PROVENANCE["human"])
    return pair[0] if mode == "light" else pair[1]


def cell_alpha(n: int) -> float:
    """Small cells fade: n=1 claims nothing and must not look like it does."""
    if n >= 20:
        return 1.0
    if n >= 5:
        return 0.55
    return 0.3


def annotate_n(n: int) -> str:
    return "" if n >= 20 else f"n={n}"


SMALL_N_FOOTNOTE = ("pale cells: n < 20; hatched: n < 5 — "
                    "not defensible individually")


def rc(mode: str) -> dict[str, object]:
    """Plain dict for matplotlib.rcParams.update(); no matplotlib import."""
    colors = palette(mode)
    return {
        "figure.facecolor": colors["surface"],
        "axes.facecolor": colors["surface"],
        "savefig.facecolor": colors["surface"],
        "text.color": colors["ink"],
        "axes.labelcolor": colors["ink_2"],
        "xtick.color": colors["ink_2"],
        "ytick.color": colors["ink_2"],
        "axes.edgecolor": colors["grid"],
        "grid.color": colors["grid"],
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "svg.hashsalt": "2ol-3alatol",  # byte-identical SVGs for unchanged data
        "figure.dpi": 110,
    }
