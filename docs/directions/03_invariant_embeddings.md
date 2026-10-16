# Direction 3 — Variant-invariant embeddings

**Status: PLANNED (follow-up)** — requires Direction 1's benchmark to exist;
ideal internship-scale continuation to propose on top of our own results.

## Problem statement

Fine-tune a multilingual sentence encoder (multilingual-e5 / LaBSE) with a
contrastive objective that pulls spelling variants of the same content
together in embedding space; evaluate retrieval@k and classification
robustness under variant noise against the un-tuned encoder.

## Why it matters

Retrieval/RAG and moderation over Arabizi break when "a7a" and "aha" land in
different embedding neighborhoods. A drop-in "franco-robust" embedding model
is immediately usable by anyone building search over Egyptian user content.

## Design

- **Positives**: (a) attested variant pairs from mining (real), (b) calibrated
  perturbations (synthetic), (c) canon pairs (x, normalize(x)).
- **Negatives**: in-batch + hard negatives from same-topic different-content
  pairs.
- **Eval**: Direction 1's retrieval benchmark + C@k; report real-vs-synthetic
  separately.

## The circularity trap (the reason this is *not* step 1)

If the perturbation engine generates both training positives and test
variants, we're teaching the model our own noise function and grading it on
the same function. Defense: test on (i) held-out *human-written* variants and
(ii) attested mined pairs never used in training. Direction 1's real-variant
set is therefore a hard prerequisite.

## MVP / ideal deliverable

MVP: e5 + kNN on Direction 1's benchmark, before/after contrastive tuning.
Ideal: released "franco-robust-embed" model + a retrieval leaderboard section.

## Agent routing

franco-benchmark-builder (train/test splits with provenance) →
franco-evaluator (before/after tables) → franco-scribe (model card).
