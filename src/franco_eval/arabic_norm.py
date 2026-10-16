"""Normalise Arabic meaning answers before comparing them.

Two model answers can name the same word and differ as strings: alef carriers
(أ إ آ vs ا), final ى vs ي, ة vs ه, stray diacritics, tatweel, or a filler
opener ("يعني ..."). Folding those is free and reproducible; anything beyond it
(synonym clustering) is a judged step that must earn its way in downstream.

The bias of every fold here is DELIBERATELY toward under-merging: every extra
fold manufactures agreement, and agreement is the quantity being measured. So
letter-run collapse is opt-in and the definite article is never stripped.

Stdlib only.
"""

import re
import unicodedata

# Arabic block letters we keep. Everything else -- Latin, digits (both ASCII
# and Arabic-Indic), punctuation, quotes -- is dropped.
_ARABIC_LETTER = re.compile(r"[ء-غف-يٱی]")

# Combining marks and tatweel removed outright.
_STRIP = dict.fromkeys(map(ord, (
    "ًٌٍَُِّْ"   # haraka + shadda/sukun
    "ٰٕٓٔ"                            # maddah, hamza marks
    "ـ"                                              # tatweel
)))

_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",  # أ إ آ -> ا
    "ٱ": "ا",                                          # ٱ -> ا
    "ى": "ي", "ی": "ي",                      # ى ی -> ي
    "ؤ": "و",                                          # ؤ -> و
    "ئ": "ي",                                          # ئ -> ي
})

_REPEAT = re.compile(r"(.)\1+")

# Filler openers a chatty model puts before the actual meaning.
FILLERS = ("يعني", "معناها", "معنى", "الكلمة", "الاجابة", "الإجابة", "هي", "هو")


def normalize_arabic(text: str, *, collapse_repeats: bool = False) -> str | None:
    """One comparable form, or None when nothing Arabic survives."""
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_STRIP).translate(_FOLD)
    # Taa marbuta folds AFTER the carrier folds and only as a letter: ة -> ه.
    # Direction matters -- اه must never grow a ة.
    text = text.replace("ة", "ه")
    # Standalone hamza carries no comparable material once carriers are folded.
    text = text.replace("ء", "")
    kept = "".join(ch if _ARABIC_LETTER.match(ch) else " " for ch in text)
    kept = " ".join(kept.split())
    if collapse_repeats:
        kept = _REPEAT.sub(r"\1", kept)
    return kept or None


def first_arabic_word(text: str, *, max_tokens: int = 3) -> str | None:
    """Extract the meaning from a possibly chatty answer.

    Strips filler openers, then: <= max_tokens remaining -> join and use them;
    more -> None. The prompt asked for one word; forcing a parse on a sentence
    would be manufacturing data, and None is a reported category, not a zero.
    """
    normalized = normalize_arabic(text)
    if normalized is None:
        return None
    tokens = normalized.split()
    fillers = {normalize_arabic(f) for f in FILLERS}
    # An opener is only filler when something follows it: a model answering
    # exactly "هو" gave a one-word answer (howa), not a preamble.
    while len(tokens) > 1 and tokens[0] in fillers:
        tokens = tokens[1:]
    if not tokens or len(tokens) > max_tokens:
        return None
    return " ".join(tokens)
