"""Corpus-facing text handling for machine-derived Arabizi.

`skeleton.words()` splits on any non-alphanumeric, which is right for the clean
hand-authored seed text it was written for and wrong for a scraped-and-
transliterated corpus. Two habits in nilechat-arabizi-egy break it:

  prime digraphs   79% of documents use an apostrophe as a diacritic -- 7' 3' 2'
                   5'. It is not applied consistently (3'eer=غير uses it as the
                   prime for غ, but e3'lam=إعلام puts it after a plain ع), so
                   there is nothing reliable to interpret. words() splits
                   et3'ayar into "et3" + "ayar", manufacturing two junk types.
  script bleed     17% of documents contain Arabic script glued to Latin
                   tokens: "felدرس dah".

Both are handled the same way: a run adjacent to an apostrophe or an Arabic
letter is not a word, and is dropped rather than guessed at. That costs ~6.6%
of tokens and removes the fragments -- `et3` falls from thousands of
occurrences to 17. We only need 500 types out of 2.3M, so there is nothing to
gain by salvaging them and an ambiguity to lose.

Stdlib only, like the rest of src/franco/.
"""

import re
from typing import Iterator

# A word is a maximal alphanumeric run that touches neither an apostrophe nor
# Arabic script on either side. The guards are lookaround rather than a
# character class so that a contaminated run is rejected whole, never trimmed
# into a shorter fake word.
_BOUNDARY = r"A-Za-z0-9'’؀-ۿ"
TOKEN_RE = re.compile(rf"(?<![{_BOUNDARY}])[A-Za-z0-9]+(?![{_BOUNDARY}])")

ARABIC_RE = re.compile(r"[؀-ۿ]")
APOSTROPHE_RE = re.compile(r"['’]")
SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")

# Sentence-initial capitals carry no information about a word being a proper
# noun, so they must not count toward cap_ratio.
_SENT_END = ".!?:\n"


def word_strings(text: str) -> list[str]:
    """Lowercased tokens only. The hot path for corpus-wide frequency counts."""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _is_sentence_start(text: str, start: int) -> bool:
    i = start - 1
    while i >= 0 and (text[i].isspace() or text[i] in "\"'([{’“"):
        if text[i] == "\n":
            return True
        i -= 1
    return i < 0 or text[i] in _SENT_END


def tokens(text: str) -> list[tuple[str, int, bool]]:
    """(lowered token, char offset, capitalised mid-sentence).

    The third field feeds the proper-noun filter: `america` is capitalised 90%
    of the time and `3ala` 0%, which separates loanwords and names from Arabizi
    without anyone curating a list.
    """
    out: list[tuple[str, int, bool]] = []
    for match in TOKEN_RE.finditer(text):
        raw = match.group(0)
        capitalised = raw[:1].isupper() and not _is_sentence_start(text, match.start())
        out.append((raw.lower(), match.start(), capitalised))
    return out


def sentences(text: str) -> Iterator[tuple[int, int, str]]:
    """(line index, sentence index within the line, sentence).

    NileChat rows are multi-line paragraphs, and a line break is a harder
    boundary than any punctuation in this corpus -- many lines are list items
    carrying no terminal mark at all.
    """
    for line_index, line in enumerate(text.split("\n")):
        for sent_index, match in enumerate(SENT_RE.finditer(line)):
            sentence = match.group(0).strip()
            if sentence:
                yield line_index, sent_index, sentence


def is_clean(text: str) -> bool:
    """No script bleed, no prime digraphs, no URLs -- safe as a carrier sentence."""
    return not (
        ARABIC_RE.search(text)
        or APOSTROPHE_RE.search(text)
        or "http" in text.lower()
    )


# Which variation a word can actually exhibit. Used to stratify the 500-word
# selection: taking the top 500 by frequency yields only 33 gemination and 58
# diphthong words, too thin to report per-phenomenon.
#
# "digit" lists 2 3 5 7 8 only -- 6 and 9 are phenomena.EXCLUDED_DIGITS, and 4
# is flagged rare there.
#
# "diph" MUST stay character-identical to the alternation of
# phenomena._DIPH_FRONT and _DIPH_BACK. src/franco/ may not import franco_eval,
# so tests/test_corpus.py asserts the equality instead of an import enforcing
# it. If they drift, the generator and the classifier disagree about what a
# diphthong is and every per-phenomenon number downstream is quietly wrong.
FEATURES: dict[str, re.Pattern[str]] = {
    "digit": re.compile(r"[23578]"),
    "diph": re.compile(r"[ae][iy](?![aeiou])|[ao][uw](?![aeiou])"),
    "vlong": re.compile(r"([aeiou])\1"),
    "gem": re.compile(r"([bdfgjklmnqrstwz])\1"),
    "digraph": re.compile(r"kh|gh|sh"),
    "finalyi": re.compile(r"[yi]$"),
}


def features(word: str) -> tuple[str, ...]:
    """Which FEATURES the word exhibits, in FEATURES order."""
    return tuple(name for name, pattern in FEATURES.items() if pattern.search(word))
