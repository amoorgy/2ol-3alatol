"""Respell several words inside one sentence, deterministically.

Dataset A's core move: take a real corpus sentence, swap some of its words for
licensed spelling variants, and emit the permutation as a training example.
The meaning is carried by the sentence; the orthography is carried by the
rules — no model participates, so same-meaning is true by construction
(modulo collisions, which the council catches downstream).

Two functions:

  respell()           pure span-splice of chosen renderings into a sentence
  sample_rendering()  seeded weighted choice of WHICH rendering each word gets

The RNG is injected (never module-level) and the sampling order is the span
order, so a fixed seed regenerates the dataset byte-identically — the repo's
hard rule 2 applied to training data.

Stdlib only.
"""

from dataclasses import dataclass
from random import Random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Span:
    """One respellable word occurrence: [start, end) in the original text."""

    word: str
    start: int
    end: int


@dataclass(frozen=True)
class Applied:
    """One substitution as it landed in the RENDERED text."""

    word: str
    rendering: str
    start: int
    end: int

    @property
    def is_base(self) -> bool:
        return self.rendering == self.word


def _validated(sentence: str, spans: Sequence[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda s: s.start)
    previous_end = -1
    for span in ordered:
        if span.start < 0 or span.end > len(sentence) or span.start >= span.end:
            raise ValueError(f"span out of bounds: {span}")
        if span.start < previous_end:
            raise ValueError(f"overlapping spans at {span.start}")
        if sentence[span.start:span.end].lower() != span.word:
            raise ValueError(
                f"span text {sentence[span.start:span.end]!r} != {span.word!r}")
        previous_end = span.end
    return ordered


def respell(sentence: str, spans: Sequence[Span],
            choices: Mapping[str, str]) -> tuple[str, tuple[Applied, ...]]:
    """Substitute each span's chosen rendering; report the shifted spans.

    Substitutions are applied RIGHT-TO-LEFT (descending start) so that earlier
    offsets stay valid while later text moves — the one subtlety of multi-span
    editing. A word absent from `choices` keeps its base spelling. The
    returned Applied offsets index into the RENDERED text, verified by tests
    to point exactly at their renderings.
    """
    ordered = _validated(sentence, spans)
    text = sentence
    shift_after: list[tuple[int, int]] = []  # (original start, cumulative delta)
    for span in reversed(ordered):
        rendering = choices.get(span.word, span.word)
        text = text[:span.start] + rendering + text[span.end:]
        shift_after.append((span.start, len(rendering) - (span.end - span.start)))

    applied: list[Applied] = []
    for span in ordered:
        rendering = choices.get(span.word, span.word)
        delta = sum(d for start, d in shift_after if start < span.start)
        applied.append(Applied(span.word, rendering,
                               span.start + delta,
                               span.start + delta + len(rendering)))
    return text, tuple(applied)


def sample_rendering(spans: Sequence[Span],
                     variants_by_word: Mapping[str, Sequence[tuple[str, float]]],
                     rng: Random, *,
                     p_respell: float = 0.55,
                     blocked: frozenset[str] = frozenset()) -> dict[str, str]:
    """One seeded permutation: per word, keep the base or pick a variant.

    `variants_by_word` maps word -> [(form, weight), ...]; weights come from
    the caller (licence tier x plausibility score), so this module stays
    agnostic about what "plausible" means.

    `blocked` is the contamination firewall: the 500 eval words and the
    human-arm words are NEVER respelled, whatever the dice say. They may still
    appear in base spelling — excluding sentences containing `elly` or `3ala`
    would delete most of the corpus — and that residual exposure is documented
    rather than hidden.

    Sampling walks spans in start order with an injected RNG: same seed, same
    spans, same variants -> byte-identical choices.
    """
    choices: dict[str, str] = {}
    for span in sorted(spans, key=lambda s: s.start):
        if span.word in choices:
            continue
        if span.word in blocked:
            continue
        candidates = list(variants_by_word.get(span.word, ()))
        if not candidates or rng.random() >= p_respell:
            continue
        forms = [form for form, _weight in candidates]
        weights = [max(weight, 1e-6) for _form, weight in candidates]
        choices[span.word] = rng.choices(forms, weights=weights, k=1)[0]
    return choices


def eligible_spans(spans: Iterable[Span],
                   variants_by_word: Mapping[str, Sequence[tuple[str, float]]],
                   blocked: frozenset[str] = frozenset()) -> list[Span]:
    """Spans that could actually be respelled — the count Dataset A gates on."""
    return [span for span in spans
            if span.word not in blocked and variants_by_word.get(span.word)]
