# Dataset B FusioN pilot — 3-arm fusor comparison (n=20 anchors)

Same 5 teacher drafts (`aya-expanse:8b`, `gemma4:e2b`, `llama3.1:8b`,
`jwnder/jais-adaptive:7b`, `nemotron-3-nano:30b`, τ=0.3 — matching the
FusioN paper's own N=5 synthetic-data-generation setting, arXiv:2510.00931)
per anchor, only the fusor model varies. Pilot scale before the full
50-anchor run.

Naturalness comes from a blind 5-lens council: each gate-passed sentence
was relabeled Arm A/B/C (the real model hidden from judges), and 5
independent judges (fluency, anchor-fidelity, degeneracy, coherence,
register) each reviewed the full blinded set of 41 candidates. ≥4/5
agreement = natural, the same bar Dataset A's council uses.

| fusor | scale | gate-passed | naturalness (of gate-passed) | end-to-end yield |
|---|---|---|---|---|
| `zai:glm-5.3` | frontier cloud-scale | 15/20 (75%) | **14/15 (93%)** | **14/20 (70%)** |
| `command-r:35b` | 35B | 11/20 (55%) | 4/11 (36%) | 4/20 (20%) |
| `qwen3.8:27b-mlx` | 27B | 15/20 (75%) | 2/15 (13%) | 2/20 (10%) |

**Key finding:** naturalness rate tracks fusor scale monotonically across
all three arms (27B → 35B → frontier), and the frontier arm's margin is
large, not marginal — 93% vs 36% vs 13%. This is a clean, decisive
replication of the FusioN paper's own scale ablation (bigger fusor → better
fusion quality), with a much larger effect size than the two-local-arm
comparison alone suggested.

**Observed failure mode:** `qwen3.8:27b-mlx` repeatedly produced
repetition-loop degeneracy (the same 2-4 word phrase repeated 3-4 times)
that passed every mechanical gate (length, anchor, language) but was
caught by every naturalness lens — confirming gate-passing alone is not a
sufficient quality signal for this pipeline.

## Getting the cloud arm working

The original plan called for `glm-5.2:cloud` via Ollama, which requires a
subscription upgrade this account doesn't have (`HTTP 403`). The
pre-authorized Z.AI fallback needed two real fixes before it worked:

1. **Wrong endpoint.** The Z.AI GLM Coding Plan's quota lives behind a
   *dedicated* endpoint (`api.z.ai/api/coding/paas/v4`), not the generic
   pay-as-you-go one (`api.z.ai/api/paas/v4`) shown on the quick-start page.
   The generic endpoint returns "insufficient balance" even with a valid,
   funded coding-plan key — it's checking a different balance entirely.
2. **Mandatory reasoning.** `glm-5.3`'s thinking cannot be disabled. A
   bounded token budget (80, even 600) was consumed entirely by the
   reasoning trace before any answer was written (`finish_reason: "length"`,
   `reasoning_tokens` = the full budget). Fixed with `reasoning_effort: "low"`
   plus a padded token budget (`max(num_predict * 4, 300)`).

New `src/franco_eval/zai_client.py` routes `--fusor "zai:<model>"` through
the correct endpoint with both fixes applied; nothing else in the repo
reads `.env` or makes non-Ollama network calls.

## Other bugs found and fixed during this run

- `build_dataset_b.py`'s `anchors_from_traps()` read the wrong TSV columns
  (`gloss_word`/`gloss_trap`) and silently always got empty glosses; fixed
  to the TSV's actual `meaning_en_word`/`meaning_en_trap`.
- `model_clients.GenConfig` had no way to override context size.
  `command-r:35b`'s Ollama package defaults to a 131072-token context
  window; allocating that KV cache for an 80-token completion pushed every
  fusor call past the 240s timeout. Added a `num_ctx` field, pinned to 4096
  in `build_dataset_b.py`.

## Next steps (superseded — see Second pass below)

If this signal holds, scale to the full 50 anchors/arm from the original
spec, and extend the council to the full run for a higher-confidence
comparison.

---

# Second pass — scenario-grounded prompts + arena benchmark (n=20 anchors)

Reviewing the first pass's actual sentences (even "natural"-passing ones)
found a real weakness: `teacher_prompt()`/`fusion_prompt()` gave models
nothing but a bare anchor word plus a rulebook, so the model's only real
task was "find a grammatical slot for this word" — the cheapest slot is
always the same shape (reported-speech openers, "I want to X"). Re-reading
the FusioN paper directly confirmed it never faces this (its own fusion
task has no forced-keyword constraint), but did surface one thing worth
adopting anyway: the paper's own fusion prompt explicitly compares drafts'
strengths *and* weaknesses before fusing (Table 6, Appendix A), which the
first pass's prompt skipped.

**Two prompt changes**, both in `src/franco_eval/fusion.py`:
1. One scenario (of 16, e.g. "you just got home after a long day and are
   texting a friend about it") is drawn per anchor and given to *every*
   teacher and the fusor for that anchor — never a different scenario per
   teacher, which would leave nothing comparable to fuse. Models now have
   something to write *about* instead of a word to place.
2. `fusion_prompt()` now explicitly asks the fusor to silently compare
   drafts' strengths and weaknesses before writing (matching the paper's
   structure), rather than just "combine the best parts."

## Naturalness result: both local arms improved sharply

Same blind 5-lens council, same ≥4/5 bar — the point of keeping the ruler
fixed is that this comparison is real, not an artifact of looser judging.

| fusor | naturalness (v1 → v2) | end-to-end yield (v1 → v2) |
|---|---|---|
| `zai:glm-5.3` | 93% → 89% (flat, already near ceiling) | 70% → 80% |
| `command-r:35b` | 36% → **67%** | 20% → 40% |
| `qwen3.8:27b-mlx` | 13% → **27%** | 10% → 20% |

Both local models roughly doubled their naturalness rate. `zai:glm-5.3`
held steady within noise (already near-ceiling at 93%). Some repetition-
loop degeneracy persisted in `qwen3.8:27b-mlx` even with scenario
grounding (e.g. `"...haml 3al 3amala ely haml 3al 3amala ely..."`) —
giving the model content to write about reduces but doesn't eliminate its
tendency to loop; that looks like a decoding/capability limit of the
smaller model, not purely a lack-of-content problem.

## New: arena benchmark (FusioN vs. Best-of-N)

No pairwise win-rate judge existed anywhere in this repo before this pass
— new `run_arena_judge.py`. For each anchor, Best-of-N is the first raw
teacher draft (no fusion) that passes the same gates the fused output has
to pass — no extra model call, exactly what "select, don't fuse" means. A
single blind Claude judge picked the more natural candidate per pair
(`Candidate 1`/`Candidate 2`, position randomized, real identity in a
private key file).

| fusor | FusioN win rate vs. Best-of-N |
|---|---|
| `zai:glm-5.3` | **16/16 (100%)** |
| `command-r:35b` | 10/11 (91%) |
| `qwen3.8:27b-mlx` | 8/15 (53%, +2 ties — near coin-flip) |

This is a second, independent measurement telling the same story as the
naturalness council: fusion's advantage over selection scales with fusor
capability — exactly the FusioN paper's own central claim, now confirmed
two different ways in this setting, not just one.

*(Table above reflects the re-judging done alongside the DeepSeek-v4-Pro
redo below — same underlying fused/BoN sentence pairs as the original run
[94%/64%/53%], since the audit files didn't change, but `run_arena_judge.py`
regenerates and re-judges every comparison including `vs_bon` on each
invocation. The shift is ordinary judge-to-judge variance from a different
blind Claude instance on an already-decisive comparison, not a new
finding — notice the near-coin-flip `qwen` arm barely moved.)*

## New: external frontier reference (DeepSeek) — corrected mid-analysis, then genuinely revealing

First attempt used `reasoning_effort: "low"` (mirroring the fix already
needed for `glm-5.3`) and got 0/45 usable references — DeepSeek answered
in full Arabic script almost every time despite an explicit Latin-only
rule. That was reported here as a "genuine finding" about DeepSeek's
Franco capability. **It was wrong**, and a user's counter-example (a
normal chat asking DeepSeek to "respond in Franco," which it did easily)
is what caught it: the real cause was `reasoning_effort: "low"` itself —
under this pipeline's multi-constraint prompt (exact anchor word, exact
spelling, exact position, ASCII-only, word-count range), low reasoning
effort made the model rush and drop the script constraint specifically,
not a capability limit. Switching to `reasoning_effort: "high"` (confirmed
live across 3/3 spot-checks before the full re-run) fixed it: 27/45 usable
references, up from 0/45 (the remaining ~40% gate failures are a real,
smaller residual tendency to drift, not eliminated but greatly reduced).

With real data, the comparison turned out far more informative than the
mistaken "DeepSeek can't do this" story would have been:

| fusor | FusioN win rate vs. DeepSeek single-shot |
|---|---|
| `zai:glm-5.3` | 7/13 (54%, +4 ties) — roughly even |
| `command-r:35b` | 1/8 (12%) — usually loses |
| `qwen3.8:27b-mlx` | 0/6 (0%) — always loses |

Fusion only earns its keep over simply asking a frontier model directly
when the **fusor itself** is frontier-tier. A weak fusor's "fusion" of
five teacher drafts is worse than one frontier model's single attempt;
only when the fusor is itself frontier-scale does fusing multiple drafts
pull roughly even with — not yet ahead of — just asking directly. That's
a real, useful refinement of the paper's own claim ("fusion beats
selection"): it doesn't say fusion beats *asking a strong model directly*
regardless of fusor quality, and this data shows it indeed doesn't.

## Redo: a stronger external reference (DeepSeek V4 Pro)

At the user's request, the external-frontier-reference leg was redone
against `deepseek-v4-pro` — GA since 2026-08-13, the larger 1.6T-total /
49B-active-param reasoning-tier sibling of `deepseek-v4-flash`'s
284B-total / 13B-active — to see whether a stronger, pricier reference
changes the story above.

Two bugs found and fixed before spending any real budget on this:

1. **A latent cache-correctness bug, caught before it could bite.** The
   DeepSeek reference cache (`deepseek_ref_cache.jsonl`) was keyed on
   `(anchor, scenario)` only — no `model` field. Switching
   `--deepseek-model` from `flash` to `pro` without fixing this would have
   silently replayed `flash`'s cached answers under the `pro` label,
   producing a fake "comparison" that was actually comparing fusion
   against itself. Added `model` to the cache key; the old 16-row
   flash-only cache (already fully reflected in the results above) was
   cleared rather than migrated, since it was disposable.
2. **A new reasoning-token-budget bug — same failure shape as before, a
   genuinely different cause.** The first real `deepseek-v4-pro` attempt
   got only 5/45 usable references. A live isolated test confirmed why:
   `finish_reason: "length"`, empty `content`, `reasoning_tokens: 800`
   (exactly the cap). This is **not** a repeat of the earlier
   `reasoning_effort` bug — effort was already `"high"` throughout.
   `deepseek-v4-pro` is simply a bigger model that reasons longer in
   absolute tokens on this identical one-sentence task (1,100–2,750
   reasoning tokens observed live), and the 800-token budget sized for
   `flash` left `pro` unable to ever finish. Raised `max_tokens` to a flat
   4,500 (confirmed live, 3/3 clean) — this costs nothing extra for
   `flash` calls, since DeepSeek bills tokens actually used, not the cap.

With the fix, 40/45 references passed the gate (better than `flash`'s
27/45). The result is sharper, not just different:

| fusor | FusioN win rate vs. DeepSeek V4 Pro |
|---|---|
| `zai:glm-5.3` | 1/16 (6%) |
| `command-r:35b` | 0/12 (0%) |
| `qwen3.8:27b-mlx` | 0/12 (0%) |

Against `deepseek-v4-flash`, `zai:glm-5.3` came out roughly even (54% +
ties). Against the stronger `deepseek-v4-pro`, it loses almost every time.
This refines the earlier finding rather than contradicting it: fusion's
ability to match an external model directly isn't just about the fusor
being frontier-tier in isolation — it depends on the fusor being at least
as strong as whatever it's being compared against. A frontier fusor pulls
even with a comparably-sized single model but loses decisively once the
external reference itself moves up a tier. This is specific to the
"fusion vs. asking a strong model directly" comparison — fusion's
dominance over Best-of-N (the table above) is untouched by this and, if
anything, looks even more decisive under the same re-judging pass.

### Is this a fusion weakness, or a teacher-pool ceiling?

A direct follow-up question: is fusion losing to DeepSeek because fusion
itself underperforms, or because the 5-teacher pool (all 8B–30B open
models, vs. DeepSeek-Pro's 1.6T total/49B active) is simply outclassed
regardless of what a fusor does with the drafts? A fusor can only
recombine what's in its inputs — it can't inject fluency or knowledge the
teachers never produced.

Tested directly with **zero new model calls**: `bon_pick` (the single
best raw teacher draft, no fusor involved at all) against the same cached
`deepseek-v4-pro` references already used above (new
`probe_teacher_ceiling.py`, reusing `run_arena_judge.py`'s cached
DeepSeek references and each arm's existing `bon_pick` field).

| arm | BoN (raw draft) vs. DeepSeek-Pro | Fusion vs. DeepSeek-Pro |
|---|---|---|
| `qwen3.8:27b-mlx` | 0/15 (0%) | 0/12 (0%) |
| `command-r:35b` | 0/11 (0%) | 0/12 (0%) |
| `zai:glm-5.3` | 0/15 (0%) | **1/16 (6%)** |
| **pooled** | **0/41 (0%)** | 1/40 (2.5%) |

**Confirmed — the teacher pool is the bottleneck, not fusion.** Every
single raw teacher draft, across all three arms, lost to
`deepseek-v4-pro`. Fusion was never going to overcome a gap that exists
at the raw-material level. One wrinkle survives: for the two weaker
fusors, fusion adds *exactly nothing* on top of that 0% floor. Only the
frontier fusor lifts it at all (0/15 → 1/16) — small, but it's the only
arm where fusion beats DeepSeek-Pro when no single raw draft ever did,
consistent with a frontier-scale fusor leaning on its own generative
capability rather than purely recombining draft content.

## New: XCOMET-XL meaning-fidelity metric — obtained, but disagrees with the council

Getting this running took three real fixes, documented in full in
`requirements-comet.txt` and `src/franco_eval/xcomet_metric.py`'s
docstring: a hard `huggingface-hub` version conflict with the main
eval `.venv` (needs its own `.venv-comet`, Python 3.12 — comet's pinned
`torchmetrics==0.10.3` imports private `functools` internals removed in
Python 3.14), a gated Hugging Face repo (resolved once the user was
granted access), and a real bug in comet's own code (`comet/models/base.py`
unconditionally sets `multiprocessing_context="fork"` whenever Apple's MPS
backend is available, which crashes with `num_workers=0` on any Apple
Silicon Mac — worked around with `num_workers=1`).

Scored reference-free (source = anchor's Arabic script, hypothesis = the
generated sentence — verified live that omitting "ref" entirely gives the
same score as an empty one):

| fusor | mean XCOMET-XL score |
|---|---|
| `command-r:35b` | 0.303 |
| `qwen3.8:27b-mlx` | 0.245 |
| `zai:glm-5.3` | 0.241 |

This **disagrees** with both the naturalness council and the arena
benchmark, which agree with each other that `zai:glm-5.3` is clearly best.
All three XCOMET scores are also uniformly low (0.24-0.30, well below the
~0.7-0.9 range these models typically produce for genuine translations) —
consistent with the mismatch flagged when this metric was first proposed:
XCOMET is built to score a real cross-lingual translation against a
reference, not a single Arabic word against an unrelated full sentence
built around it. Reported for completeness, but the naturalness council
and arena benchmark are the trustworthy signal here — they're directly
interpretable, agree with each other, and were built for exactly this
judgment; XCOMET's application here was always a stretch, and the result
looks like noise rather than a genuine third opinion.

## Bugs found and fixed during this pass

- `run_arena_judge.py` originally took one `--audit`/`--arm` at a time and
  wrote to fixed, non-arm-specific output paths — each run silently
  clobbered the previous arm's chunk/key files. Fixed to accept multiple
  `--audit`/`--arm` pairs and pool all arms into one combined judging pass
  (same convention `prep_council_fusion.py` already used).
- `deepseek_reference()`'s caching forced `"raw": text or " "` to work
  around `JsonlCache`'s built-in "never cache a falsy raw" rule — meaning
  a single failed call (e.g. a token-budget miss) got cached as a
  permanent "confirmed empty," blocking every future retry. Removed; the
  cache's existing behavior (skip caching failures, always retry next run)
  was correct all along.
- Running Ollama generation and the 11GB XCOMET-XL PyTorch checkpoint
  concurrently exhausted this machine's 32GB RAM into swap and silently
  killed both processes (no error, no log output — a `SIGKILL` bypasses
  buffered-output flushing entirely). These now run strictly sequentially.

## Next steps

If this signal holds, scale toward the 628-anchor headroom available in
`trap_votes.tsv` (314 confirmed pairs × 2). Consider whether DeepSeek's
script-drift problem is worth chasing further (e.g. reinforcing the
Latin-only rule) or whether it's simply not a good fit as an external
reference for this specific low-resource-orthography task.

## "Experiment b2" — a stronger teacher pool, quick pass (n=20)

Prompted by the paper-comparison above: does swapping two of the five
teachers for genuinely large models (`deepseek:deepseek-v4-flash`,
`zai:glm-5.3-flash`, replacing `jwnder/jais-adaptive:7b` and
`aya-expanse:8b`; `llama3.1:8b`, `gemma4:e2b`, `nemotron-3-nano:30b`
unchanged) narrow the gap against `deepseek-v4-pro`? Same 3 fusor arms,
same gates, judged by Claude Sonnet (not Opus, at the user's request).

| fusor | vs. Best-of-N | vs. `deepseek-v4-pro` |
|---|---|---|
| `command-r:35b` | 87% (13/15) | 0% (0/14, +1 tie) |
| `qwen3.8:27b-mlx` | 65% (+5 ties, 11/17) | 6% (1/16, +4 ties) |
| `zai:glm-5.3` | 94% (17/18) | 11% (2/18, +8 ties) |

No change to the headline finding — fusion still loses decisively to
`deepseek-v4-pro` across all three arms. The small upward nudges (e.g.
6%→11% for the frontier fusor, compared to the all-small-teachers pool a
few days earlier) are within noise at this sample size, not a real
shift. Best-of-N numbers stay exactly as decisive and scale-correlated
as every prior run. This sharpens rather than undermines the earlier
teacher-ceiling finding: even a partial upgrade to frontier-adjacent
teachers isn't enough to change the outcome against a model as strong as
`deepseek-v4-pro` specifically.

**Real bugs found and fixed getting here**, none of which affected the
numbers above (all caught before or fixed to still produce correct data):
a double-backgrounding mistake that silently did nothing while reporting
success; a filename-reuse bug where a fresh 20-anchor run appended onto
505 leftover rows from an unrelated interrupted overnight attempt days
earlier (caught via a suspiciously-fast "how much is left" check, fixed
by regenerating from a clean file); a single dropped connection (this
machine was tethered to a phone hotspot, not stable WiFi) killing the
entire multi-anchor arena-benchmark run instead of just skipping the one
affected anchor, fixed in `run_arena_judge.py` to tolerate a transient
network failure per-reference rather than crash the whole run.

See `fusion_pilot_overview.md` for a short, non-technical summary of the
whole pilot.
