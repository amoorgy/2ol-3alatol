# Meaning metrics — scoring reference

How `franco_eval.meaning_metrics.score()` turns parsed model answers into the
Level 3 table in [`level3_meaning.md`](level3_meaning.md). This is a reference
for the scoring function itself, not a new experiment — see `level3_meaning.md`
for the measured numbers and `README.md` for how Level 3 fits the other levels.

## Data flow

`ItemMeanings` (one probe item: base + variants + optional trap + optional
repeat) → canonicalize meaning strings to ints → count calls/parses →
`score()` computes five independent metric families → a flat dict of 18
metrics, never combined into one score.

## The core design: 2×2 failure modes

- **Variants of one word SHOULD agree.** Disagreement = *under-robustness*.
- **A trap (different word) SHOULD differ.** Agreement = *over-normalization*.

These are never collapsed — each has its own metric and its own denominator.

**The None asymmetry is deliberate.** For variants, a non-answer counts
*against* agreement: failing to produce a meaning is a failure to demonstrate
robustness. For traps, None is excluded from the denominator
(`n_traps_scored`): a refusal is *no data*, not evidence the model detected
the trap — and it is reported separately as `trap_non_answer_rate`.

## Metric families

### Variant robustness — do synonyms agree?

| Metric | Formula | Direction |
|---|---|---|
| `variant_pairwise_agreement` | matches / total pairs, compared via int ids | higher is better |
| `under_robust_rate` | `1 - agreement` | lower is better |
| `variant_consistency_at_k` | items where all k renderings match / n_items | strict; floor note, not headline |
| `modal_meaning_share` | avg. `count_modal / k` per item, k = 1 + n_variants | consensus without requiring unanimity |

### Trap detection — do false cognates get caught?

| Metric | Formula | Direction |
|---|---|---|
| `over_norm_rate` | `n_over / n_traps_scored` (traps whose meaning equals the modal base answer) | lower is better |
| `trap_non_answer_rate` | `n_trap_none / n_trap_items` | informational — was the trap even attempted |

### Generation stability ceiling

| Metric | Formula | Direction |
|---|---|---|
| `ceiling_agreement` | `n_repeat_agree / n_repeat_scored` (base asked twice) | upper bound on variant agreement — the noise floor |
| `ceiling_n` | items with a scoreable repeat (excludes double non-answers) | denominator for the above |

### Gold accuracy — where ground truth exists

| Metric | Formula | Direction |
|---|---|---|
| `meaning_accuracy` | `n_gold_correct / n_gold`, per-meaning not per-item | only counted where `item.gold` is non-empty |

### Per-phenomenon flip rates (`by_class`)

| Metric | Formula | Direction |
|---|---|---|
| `flip_rate` | `n_flips / n_edges` per `ortho_class` | identifies which spelling phenomena cost robustness |

An edge = one variant + its base comparison; `n_none` tracks edges where
either side was None, counted separately from `n_flips`.

## Why nothing is collapsed into one score

- Under-robustness and over-normalization are different problems needing
  different interventions.
- The ceiling is a noise floor — how much it should count depends on use case.
- Per-phenomenon rates reveal root causes that a global number would hide.
- Gold accuracy only makes sense where ground truth exists at all.

| Family | Input data | Denominator | None handling | Weight relationship |
|---|---|---|---|---|
| Variant agreement | base + variants (all items) | all pairs compared | None ≠ None (disagree) | symmetric — each variant weighted equally |
| Trap detection | trap vs. modal base | only traps with modal base ≠ None | None → excluded from denominator | directed — trap vs. one canonical answer |
| Ceiling | base vs. base_repeat | items with ≥1 non-None | double None → excluded (uninformative) | stability only, no weighting with variants |
| Gold | all meanings vs. gold set | only where `item.gold` ≠ empty | None → incorrect (not scored) | external truth, independent denominator |
| Per-phenomenon | variant meanings by class | all edges in that class | None tracked separately (`n_none`) | per-class, never aggregated |

`n_calls` includes every call (base + variants + traps + repeats), but
agreement only uses base + variants — traps and ceiling are measured
separately on purpose. `non_answer_rate = (n_calls - n_parsed) / n_calls` is
the proxy for "how often the pipeline failed to return a meaning at all"
(parse failures, wrong-script detections, API errors combined).

## When metrics are `None`

| Metric | `None` when |
|---|---|
| `variant_pairwise_agreement` | `int_groups` is empty (no base + variants data) |
| `over_norm_rate` | `n_traps_scored == 0` (all traps failed to parse, or no traps) |
| `ceiling_agreement` | `n_repeat_scored == 0` (no scoreable repeats) |
| `meaning_accuracy` | `n_gold == 0` (no gold ground truth) |

`None` is a value here, not a gap — the Level 3 NileChat arm carries no gold
Arabic, so `meaning_accuracy` is `n/a` by design for every model.

## Worked examples

**Perfect robustness** — model returns the same meaning for base and variant
on every item:

```
variant_pairwise_agreement: 1.0
under_robust_rate:          0.0
consistency_at_k:           1.0
modal_meaning_share:        1.0
```

**Over-normalization (false cognate)** — base "قعل"→"act", variant "قال"→
"said", trap "قعيل" wrongly answered "said" too:

```
modal_meaning(item) = "said"      (appears 2x: base + variant)
n_traps_scored       = 1          (trap answer is not None)
n_over               += 1         (trap == modal meaning)
over_norm_rate        = 1/1 = 1.0   <- FAILURE
```

**Generation instability** — base asked twice, "act" then "beat":

```
n_repeat_scored  = 1   (both answers are not None)
n_repeat_agree   = 0   (base "act" != base_repeat "beat")
ceiling_agreement = 0/1 = 0.0   <- UNSTABLE
```

**Per-phenomenon analysis** — one DOUBLING edge agrees, one VOWEL_DROP edge
disagrees:

```
by_class["DOUBLING"]    = {n_edges: 1, n_flips: 0, flip_rate: 0.0}
by_class["VOWEL_DROP"]  = {n_edges: 1, n_flips: 1, flip_rate: 1.0}
```

Source: [`src/franco_eval/meaning_metrics.py`](../../../src/franco_eval/meaning_metrics.py).
Measured results from this scoring function: [`level3_meaning.md`](level3_meaning.md).
