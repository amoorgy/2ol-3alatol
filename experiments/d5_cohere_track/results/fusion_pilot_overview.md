# FusioN pilot — overview

**Question:** does fusing 5 teacher-model drafts into one sentence (the FusioN
technique, arXiv:2510.00931) produce better synthetic Egyptian-Arabizi
training data than simpler alternatives — picking one draft (Best-of-N), or
just asking one good model directly?

## What I tried

1. **3-arm fusor comparison** (`qwen3.8:27b-mlx` 27B, `command-r:35b` 35B,
   `zai:glm-5.3` 743B) fusing the same 5 teacher drafts, judged for
   naturalness by a blind 5-lens council and for pairwise preference by a
   blind arena benchmark (fusion vs. Best-of-N, fusion vs. an external
   single-shot model).
2. When fusion lost to the external model (DeepSeek) almost every time, I
   checked *why*: compared even the single best raw teacher draft — no
   fusion at all — against DeepSeek. It also lost, 0 wins out of 41.
3. Checked the numbers against the FusioN paper's own setup: their teacher
   pool ranged 27B up to 1T params, already comparable in scale to their
   external reference (Gemini). Ours was capped at 30B — a ~50x gap versus
   `deepseek-v4-pro`. The paper never tested the regime we were testing.
4. Redesigned the teacher pool to close some of that gap: swapped two of
   the five teachers for larger models (`deepseek-v4-flash`, `glm-5.3-flash`),
   re-ran the same 3 fusor arms against `deepseek-v4-pro`.

## What worked

- **Fusion reliably beats Best-of-N** (picking your own best draft),
  scaling with fusor size, across every configuration tested: 53-100%
  win rate depending on fusor scale. This replicates the FusioN paper's
  own central claim cleanly.
- **Naturalness improved sharply** once teacher/fusor prompts were given
  a concrete scenario to write about instead of a bare word to place
  (both local fusors roughly doubled their naturalness pass rate).
- The diagnostic chain itself worked: each surprising result (fusion
  losing to DeepSeek, DeepSeek "failing" at Franco) was investigated to
  a real, verifiable cause rather than accepted at face value — one of
  those investigations (a reasoning-effort setting causing DeepSeek to
  drop script constraints under load) was a genuine bug fix, not a
  finding about the model's actual capability.

## What didn't work

- **Fusion does not beat asking a strong external model directly**, in
  any tested configuration. Best case: a rough tie, only with the most
  expensive fusor, only against a comparably-sized reference. Every other
  pairing loses outright, including after upgrading two of five teachers
  to genuinely large models — that upgrade nudged the numbers slightly
  (6%→11% for the strongest fusor) but didn't change the outcome.
- Free-tier API routes (OpenRouter's free Nvidia/Nemotron model) proved
  unreliable in production — three different failure modes across
  attempts — and were dropped in favor of already-proven paid/local
  routes rather than chased further.

## Bottom line

Fusion is a real, measurable improvement over naive draft-selection, and
that holds at every scale tested. But it is not a way to make a pool of
small-to-mid models collectively outperform a single frontier model — the
raw material ceiling set by the teacher pool bounds what any fusor can
produce, and no fusor tested (up to 743B) could fuse its way past that
ceiling against `deepseek-v4-pro` specifically. For practical synthetic
data generation at this scale, the data supports picking one good model
and prompting it directly over building a multi-teacher fusion pipeline —
unless the teacher pool itself is upgraded to genuinely frontier scale
across the board, which is a materially bigger undertaking than the one
tested here.
