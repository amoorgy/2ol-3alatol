# Direction 1 — "2ol-tany": orthographic-robustness benchmark + pipeline bake-off

**Status: ACTIVE — showcase running with real results** (`results/showcase.md`).
This is the spine of the project; Directions 2–4 plug into it.

## Problem statement

Given the same Egyptian Arabizi content realized in k plausible spelling
variants, measure (i) prediction consistency and (ii) accuracy degradation of
current systems, and determine which input-handling strategy — raw subword,
light normalization, transliteration-to-Arabic-script, byte-level — is most
robust under matched budgets.

## Why it matters

Arabizi is the daily written register of tens of millions of Egyptians; models
demonstrably behave differently across spellings of the same content (the
Arabizi-jailbreak result, EMNLP 2024). No benchmark measures it; no controlled
pipeline comparison exists (verified against mid-2026 literature).

## What exists in this repo now

| Piece | File | State |
|---|---|---|
| Skeleton key (variant fingerprint) | `src/franco/skeleton.py` | done, tested (morphology-aware: final -y/-i kept) |
| Variation taxonomy (31 weighted rules) | `src/franco/taxonomy.py` | v0 — needs native-speaker weight review |
| Perturbation engine (compound, seeded) | `src/franco/perturb.py` | done, tested |
| Franco canon v0 (normalizer) | `src/franco/normalize.py` | done, tested — NOTE: inverts part of the taxonomy; synthetic-eval leakage is measured and reported |
| Variant miner (affix-guarded) | `src/franco/mine.py` | done — 57 attested clusters from seed corpus |
| Retrieval baseline + metrics | `src/franco/retrieval.py`, `metrics.py` | done, tested (C@k counts total failure as inconsistent) |
| Showcase experiment | `experiments/d1_benchmark/run_showcase.py` | done — 5-seed mean, clean control, leakage diagnostic, attested-variant table |
| Benchmark artifact v0 | `data/benchmark/retrieval_v0.jsonl` | generated (synthetic); attested swaps evaluated in-run |

## Ladder of next steps (each one is a visible increment)

1. **Native-speaker calibration** — review taxonomy weights against
   `data/benchmark/variant_clusters.tsv` (owner: you + franco-perturber).
2. **Real-variant held-out set** — hand-write ~150 authentic variants;
   evaluate whether system rankings hold on human noise vs synthetic
   (the benchmark's central methodological defense).
3. **Labeled classification task** — ~500 messages, 3-way sentiment,
   double-annotate 150, report Cohen's κ (franco-benchmark-builder).
4. **Wave-1 model arms** — char-n-gram SVM, XLM-R fine-tune, ByT5-small
   fine-tune; report accuracy + C@k per intensity (franco-evaluator).
5. **Wave-2 arms** — transliteration front-end (Direction 2), NileChat-3B /
   Nile-Chat-4B few-shot, embedding retrieval; the augmentation ablation
   (train on perturbed data — does the cheapest fix close the gap?).
6. **Package** — HF dataset + datasheet + 4-page report.

## Evaluation contract

recall@1 / macro-F1 (clean + per-variant) · C@k · worst-variant gap ·
degradation slope over intensity · bootstrap 95% CIs · real-vs-synthetic split
reported separately · every number reproducible from a committed script + seed.

## Failure modes being managed

Synthetic variants unlike real ones (→ step 2); task saturation (→ fragment
queries, already hit and fixed once); leakage via near-duplicates (→ dedupe by
skeleton sequence); contamination (→ no fineweb-edu-Egypt-derived text).
