"""Canonical Romanized Egyptian form — "Franco canon v0".

No standard orthography for Arabizi exists (CODA* covers only the
Arabic-script side), so this module *defines* a minimal one:

  - unambiguous Arabic phonemes prefer the digit:  kh→5, gh→8, q→2
  - plain h is left alone (ه vs ح is unrecoverable without a lexicon —
    this documented ambiguity is a real limit of normalization, and it
    should show up in the results)
  - ch→sh; vowels unify to the Egyptian defaults i→e, u→o
  - any run of a repeated character collapses to one (kills both
    gemination spelling and elongation, lossy on purpose)
  - final -ah → -a (taa marbuta)
  - lowercase, whitespace collapsed

Lossy by design: *consistent* is the goal, not faithful.
Idempotent: normalize(normalize(x)) == normalize(x).
"""

import re

_STEPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ch"), "sh"),
    (re.compile(r"kh"), "5"),
    (re.compile(r"gh"), "8"),
    (re.compile(r"q"), "2"),
    (re.compile(r"i"), "e"),
    (re.compile(r"u"), "o"),
    (re.compile(r"(.)\1+"), r"\1"),
    (re.compile(r"ah\b"), "a"),
    (re.compile(r"\s+"), " "),
)


def normalize(text: str) -> str:
    out = text.lower().strip()
    for rx, repl in _STEPS:
        out = rx.sub(repl, out)
    return out
