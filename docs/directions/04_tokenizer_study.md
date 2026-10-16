# Direction 4 — Tokenizer segmentation-instability study

**Status: PLANNED (analysis chapter)** — ~3 days of work, no model training;
lives inside Direction 1's report as its mechanism-explaining figure.

## Problem statement

Quantify, across major tokenizers (Gemma, Llama, Qwen, XLM-R, mT5 + an
Arabic-script tokenizer post-transliteration), how token fertility and
segmentation boundaries diverge across spelling variants of the same word
("a7a" → `a·7·a` vs "aha" → `aha`), and whether that instability predicts
downstream prediction flips.

## Why it matters

Extends the tokenizer-fairness literature (Petrov et al. 2023 — up to 15x
cross-language cost differences) to the unstudied case of *intra-language*
orthographic variation, where digit intrusions break BPE merges in a
characteristic way. If instability correlates with flips, tokenizer-aware
fixes become a motivated follow-up; if not, that's an equally citable
negative result.

## Metrics

- **Fertility ratio**: tokens-per-word for each variant vs its cluster head.
- **Boundary divergence**: 1 − Jaccard overlap of token-boundary character
  offsets between variant pairs.
- **Flip correlation**: point-biserial correlation between per-item boundary
  divergence and whether the model's prediction flipped between the variants.

## Inputs it needs (all already exist)

Variant pairs: `data/benchmark/variant_clusters.tsv` (attested) +
`data/benchmark/retrieval_v0.jsonl` (synthetic). Tokenizers via HF
`transformers` on Colab — the one part not runnable stdlib-only.

## MVP / ideal deliverable

MVP: fertility + divergence table across 6 tokenizers on 1k variant pairs.
Ideal: the flip-correlation figure inside Direction 1's report.

## Explicit non-goal

**No tokenizer training.** This direction produces evidence, not engineering.
Only if instability strongly predicts flips AND byte-level closes the gap at
unacceptable cost does tokenizer engineering become a (separate, future)
proposal — grounded in these measurements.

## Agent routing

franco-evaluator (all of it) → franco-scribe (figure + section).
