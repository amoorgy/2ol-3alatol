"""Egyptian Franco orthographic variation taxonomy.

Each rule is one attested spelling habit, written as a regex rewrite with a
relative plausibility weight (higher = more common in the wild). The weights
drive sampling in perturb.py, so they shape every benchmark number downstream.

=======================================================================
NATIVE-SPEAKER REVIEW NEEDED (this is a scientific input, not cosmetics)
Adjust the weights below from your own intuition of what Egyptians
actually type, and add habits that are missing. Calibration against
mined real variant pairs (mine.py output) is the week-1 milestone.
=======================================================================
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    pattern: str      # regex searched inside a single word
    replacement: str  # may use backrefs like \1
    weight: float     # relative plausibility, any positive scale
    phenomenon: str   # taxonomy label used in analyses


RULES: tuple[Rule, ...] = (
    # --- digit <-> letter substitutions --------------------------------
    # Guarded on both sides: s7→sh / k7→kh / g7→gh / c7→ch would manufacture
    # digraphs that read as different phonemes (ش/خ/غ), which writers avoid.
    Rule(r"(?<![skgc])7", "h", 0.9, "digit-to-letter 7→h (ح)"),
    Rule(r"(?<![skgtc])h", "7", 0.5, "letter-to-digit h→7 (ambiguous ه/ح)"),
    Rule(r"kh", "5", 0.7, "digraph-to-digit kh→5 (خ)"),
    Rule(r"5", "kh", 0.7, "digit-to-digraph 5→kh (خ)"),
    Rule(r"gh", "8", 0.4, "digraph-to-digit gh→8 (غ)"),
    Rule(r"8", "gh", 0.4, "digit-to-digraph 8→gh (غ)"),
    Rule(r"9", "s", 0.4, "digit-to-letter 9→s (ص)"),
    Rule(r"6", "t", 0.4, "digit-to-letter 6→t (ط)"),
    Rule(r"3", "a", 0.3, "ع weakened to plain vowel"),
    Rule(r"2", "a", 0.5, "hamza dropped to plain vowel"),
    Rule(r"q", "2", 0.6, "qaf written as glottal stop"),
    # --- consonant spelling habits --------------------------------------
    Rule(r"sh", "ch", 0.3, "sh→ch (francophone habit)"),
    Rule(r"g", "j", 0.15, "gim as j (loanword / non-Cairene habit)"),
    # --- vowel habits ----------------------------------------------------
    Rule(r"e", "i", 0.6, "kasra vowel swap e→i"),
    Rule(r"i", "e", 0.6, "kasra vowel swap i→e"),
    # Word-final only. Unanchored these fire mid-word (ya3ny -> ia3ny),
    # which is not a spelling habit anyone has; the y/i alternation that
    # Egyptians do type is the final one (enty/enti, ya3ny/ya3ni).
    Rule(r"y\b", "i", 0.5, "final y→i vowel spelling (enty/enti)"),
    Rule(r"(?<=[a-z0-9])i\b", "y", 0.25, "final i→y vowel spelling (ya3ni/ya3ny)"),
    Rule(r"o", "u", 0.5, "damma vowel swap o→u"),
    Rule(r"u", "o", 0.5, "damma vowel swap u→o"),
    Rule(r"([aeiou])\1", r"\1", 0.5, "long vowel shortened"),
    Rule(r"([aeiou])", r"\1\1", 0.2, "vowel elongation (emphasis)"),
    Rule(r"(?<=[a-z0-9])[aeiou](?=[a-z0-9])", "", 0.3, "medial vowel deletion"),
    # --- diphthong habits -------------------------------------------------
    # /aj/ and /aw/ nuclei alternate with their monophthongs: 3ein/3een/3en,
    # beit/beet/bet, yom/youm. The (?![aeiou]) lookahead keeps these off onset
    # glides -- in `tayeb` the y starts the next syllable and must not fold.
    Rule(r"ay(?![aeiou])", "ei", 0.4, "diphthong spelling ay→ei (زي)"),
    Rule(r"ay(?![aeiou])", "ee", 0.5, "monophthongisation ay→ee (بيت)"),
    Rule(r"ei(?![aeiou])", "ee", 0.6, "monophthongisation ei→ee (عين)"),
    Rule(r"ee", "ei", 0.3, "diphthong spelling ee→ei"),
    Rule(r"aw(?![aeiou])", "oo", 0.4, "monophthongisation aw→oo (لون)"),
    Rule(r"ou(?![aeiou])", "oo", 0.5, "monophthongisation ou→oo (يوم)"),
    # --- word-shape habits -----------------------------------------------
    Rule(r"([b-df-hj-np-tv-z])\1", r"\1", 0.5, "degemination (double→single)"),
    Rule(r"ah\b", "a", 0.3, "final -ah→-a (taa marbuta)"),
    Rule(r"a\b", "ah", 0.15, "final -a→-ah (taa marbuta)"),
)

COMPILED: tuple[tuple[re.Pattern[str], Rule], ...] = tuple(
    (re.compile(rule.pattern), rule) for rule in RULES
)
