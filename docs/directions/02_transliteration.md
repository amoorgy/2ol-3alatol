# Direction 2 — Free Egyptian Arabizi→Arabic transliterator

**Status: PLANNED** — becomes one arm of Direction 1's bake-off first; a
standalone deliverable only if the arm shows promise.

## Problem statement

Train a lightweight (ByT5-small-class) Arabizi→Arabic-script transliterator
*without* the paid LDC gold corpus, using reverse-romanization of free
Egyptian-script corpora with sampled orthographic variation plus a small gold
set we annotate ourselves; measure CER/word accuracy and the downstream gain
when prepended to Arabic-script models (MARBERT / CAMeLBERT-DA).

## Why it matters

The only strong Egyptian transliterators (Shazal/Usman/Habash 2020: 80.6% word
accuracy) train on LDC2017T07 (paid). The sole open Franco→Arabic model on HF
is an undocumented personal T5-small. A free, reproducible recipe would be
widely used.

## Recipe

1. **Reverse-romanizer**: rule-based Arabic-script→Arabizi generator reusing
   `src/franco/taxonomy.py` weights in reverse — every Arabic word yields
   several plausible Franco spellings. Source corpora (all verified free):
   ETC (34M Egyptian tweets, CC-BY-4.0), ASTD (10k, labels carry through!),
   Habibi Egyptian lyrics.
2. **Train** ByT5-small on ~50k synthetic pairs (Colab T4 scale).
3. **Gold test set**: 200 sentences hand-transliterated (CODA-guided, checked
   with `camel_tools`) — the only manual cost.
4. **Evaluate**: CER + word accuracy on gold; then downstream delta inside
   Direction 1's bake-off, reporting transliterator error separately from
   downstream error (attribution!). Include an oracle-transliteration ceiling.

## MVP / ideal deliverable

MVP: reverse-romanizer + ByT5-small + CER on gold. Ideal: released model +
generator + gold set + downstream-delta table.

## Risks

Synthetic-to-real romanization-style gap (mitigate: mined clusters inform the
reverse-romanizer's weights); comparison against LDC-trained numbers we cannot
reproduce (state it, don't hide it).

## Agent routing

franco-miner (source corpora) → franco-perturber (reverse-romanizer rules) →
franco-evaluator (CER + downstream) → franco-scribe (model card).
