"""Which script is an answer written in?

The meaning probe demands an Arabic-script answer. A Latin or mixed reply is a
failure to follow the instruction, not a meaning, and it must land in the
non-answer category rather than being compared as a string.

Stdlib only.
"""

import re

_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN = re.compile(r"[A-Za-z]")


def arabic_ratio(text: str) -> float:
    """Arabic letters / (Arabic + Latin letters).

    Digits and punctuation sit outside the denominator on purpose: "عايز."
    is a fully Arabic answer, and an Arabic meaning followed by "(:" must not
    be penalised for the smiley.
    """
    arabic = len(_ARABIC.findall(text))
    latin = len(_LATIN.findall(text))
    if arabic + latin == 0:
        return 0.0
    return arabic / (arabic + latin)


def is_arabic_answer(text: str, threshold: float = 0.8) -> bool:
    return bool(text) and arabic_ratio(text) >= threshold
