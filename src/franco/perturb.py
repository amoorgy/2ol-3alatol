"""Calibrated spelling-variant generator.

Given a Franco text, rewrites each word with probability `intensity` by
sampling one applicable taxonomy rule (weight-proportional) and applying it
at one random match position. Deterministic given the same random.Random.
"""

import random
import re

from .taxonomy import COMPILED, Rule

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_MIN_WORD_LEN = 2  # 1-char tokens ("w") yield garbage; 2+ participates
                   # (function words el/ya/eh carry heavy real variation)


def _applicable(word: str) -> list[tuple[re.Pattern[str], Rule]]:
    return [(rx, rule) for rx, rule in COMPILED if rx.search(word)]


def perturb_word(word: str, rng: random.Random, depth: int = 1) -> str:
    """Up to `depth` stacked rewrites of one word; unchanged if no rule applies.

    Case-folds first: taxonomy rules are lowercase, and case is not part of
    the variation under study — variants come out lowercase by design.
    """
    word = word.lower()
    for _ in range(depth):
        options = _applicable(word)
        if not options:
            break
        rx, rule = rng.choices(options, weights=[r.weight for _, r in options])[0]
        matches = list(rx.finditer(word))
        m = rng.choice(matches)
        word = word[: m.start()] + m.expand(rule.replacement) + word[m.end() :]
    return word


def perturb(text: str, intensity: float, rng: random.Random, depth: int = 3) -> str:
    """Rewrite each word of `text` with probability `intensity`.

    Rewritten words receive 1..depth stacked rule applications — real Franco
    variation is compound ("ezzayyak"→"izayak" differs by three habits), so a
    single edit per word understates it.
    """

    def _sub(m: re.Match[str]) -> str:
        word = m.group(0)
        if len(word) >= _MIN_WORD_LEN and rng.random() < intensity:
            return perturb_word(word, rng, rng.randint(1, depth))
        return word

    return _TOKEN_RE.sub(_sub, text)


def variants(
    text: str,
    k: int,
    intensity: float,
    rng: random.Random,
    max_tries: int = 50,
) -> list[str]:
    """Up to k distinct perturbed renderings of `text` (never `text` itself)."""
    out: list[str] = []
    seen = {text}
    for _ in range(max_tries):
        if len(out) == k:
            break
        v = perturb(text, intensity, rng)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
