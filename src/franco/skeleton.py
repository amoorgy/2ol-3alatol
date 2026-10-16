"""Phonemic skeleton key for Egyptian Franco (Arabizi).

Maps every plausible spelling of a word to one consonant fingerprint, so
"a7a"/"aha", "2alb"/"alb"/"qalb", and "khalas"/"5alas" all collide.

Like Soundex, it deliberately over-merges: it produces *candidate* variant
clusters, and the miner (mine.py) confirms them with edit distance.
"""

import re

# Two Latin letters standing for one Arabic phoneme; replaced first.
_DIGRAPHS = (
    ("kh", "x"),  # خ
    ("gh", "G"),  # غ
    ("sh", "S"),  # ش
    ("ch", "S"),  # ش (francophone chat spelling)
)

# Franco digits and their phoneme classes.
_DIGITS = {
    "2": "q",  # ء — and Cairene ق, which is why plain q joins this class
    "3": "E",  # ع
    "5": "x",  # خ
    "6": "t",  # ط (merges with ت on purpose: writers interchange them)
    "7": "h",  # ح (merges with ه on purpose: writers interchange 7/h)
    "8": "G",  # غ
    "9": "s",  # ص (merges with س on purpose)
}

# y/i count as vowels word-medially ("3ayez"/"3aiz" must collide) but are
# KEPT word-finally: final -y/-i is a morpheme (enty 2fs, masry nisba,
# habiby possessive) — stripping it would merge whole paradigms
# (enta/enty are different words, not spelling variants).
_VOWELS = frozenset("aeiouy")

_WORD_RE = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    """Lowercased Franco word tokens (letters+digits) of a text."""
    return _WORD_RE.findall(text.lower())


def _collapse(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def skeleton(word: str) -> str:
    """Phonemic fingerprint of one word. Empty input returns ''."""
    w = word.lower()
    for src, dst in _DIGRAPHS:
        w = w.replace(src, dst)
    w = "".join(_DIGITS.get(ch, ch) for ch in w)
    # A WORD-INITIAL glottal stop is optional in writing ("2alb"/"alb"/"qalb")
    # — decide before vowel removal so word-medial ق/ء ("a2ollak") survive.
    if w[:1] == "q":
        w = w[1:]
    last = len(w) - 1
    core = "".join(
        ch
        for i, ch in enumerate(w)
        if ch not in _VOWELS or (i == last and ch in "iy")
    )
    # Final -y and -i spell the same morpheme: unify to y ("enty"/"enti").
    if core.endswith("i"):
        core = core[:-1] + "y"
    core = _collapse(core)
    if not core:
        # Vowel-only words ("eeeh") fall back to the collapsed vowel string
        # so they still cluster.
        core = _collapse("".join(ch for ch in w if ch in _VOWELS))
    return core
