# Dataset v2 — Egyptian Arabizi spelling robustness (fine-tuning)

Teaches two rules at once: different spellings of one word mean the same
thing (SAME), and some look-alike spellings are different words (DIFFERENT).
Built 2026-09-23 → 09-26; exact counts, file hashes and seeds in `stats.json`.

## Files

Row files are gitignored (too large) and regenerate from tracked inputs:
`data/generated/v2/final/`.

| file | train | val | row type |
|---|---:|---:|---|
| `mixed_{train,val}.jsonl` | 269,306 | 29,904 | both types, deterministically shuffled (point a trainer here) |
| `normalize_{train,val}.jsonl` | 267,579 | 29,719 | user: sentence in mixed spellings → assistant: base spelling |
| `meaning_{train,val}.jsonl` | 1,727 | 185 | user: sentence + one word → assistant: that word in Arabic script |

Chat JSONL: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "meta": {...}}`.
`meta` carries `task`, `rendering_id`, `source`, `provenance`. Instruction
wording rotates over 10 (normalize) / 11 (meaning) templates in English,
Arabic and Arabizi, none of them the Level-3 probe's wording.
`sample.jsonl` (tracked) holds 200 mixed rows to eyeball.

The per-task files exist so a trainer can re-weight: normalize outnumbers
meaning about 155 to 1.

## Evaluation — read before using

The benchmark's words were split (`data/stimulus/nilechat/word_split.tsv`):
225 train-side words and their traps ARE respelled, anchored and asked about
in this data; 275 held-out words, their traps and every human-arm word never
are (no held-out form is ever respelled or created, and none is a fusion
anchor). They do still occur, unchanged, as ordinary context words in most
rows. **Evaluate only on the held-out side** (Level 1/3 probes on those items). Scores on train-side words measure
memorisation, not the rule. Train-side meaning rows share the Level-3 task
but not its prompts.

## Sources and provenance

| source | provenance | rows | how |
|---|---|---:|---|
| NileChat Arabizi-EGY | `nilechat_machine_derived` (Command R+ transliteration) | 296,024 normalize | 152,707 sentences respelled by licensed rules — no model writes a spelling (158,452 collected from all 571,524 docs; 5,745 had < 2 usable words) |
| Fusion | `fusion_synthetic` | 1,274 normalize + 1,912 meaning | 5 teachers (llama3.1:8b, gemma4:e2b, nemotron-3-nano:30b, deepseek-v4-flash, glm-5.3-flash) → fusor glm-5.3, around 298 trap anchors × 4 scenarios |

Both sources are machine-derived (hard rule 1): never a benchmark artifact.

## Verification

1. **Substitution policy** (`src/franco_eval/substitution_policy.py`): the
   a_v1 review's rulings and the native rulings of 2026-09-25 as code.
   Rules-only residual error measured on 5,400 judged a_v1 rows: 6.1% (from
   17.9%). Judging the flagged pairs (below) removes part of that residual.
2. **Pair council** on the 6,804 substitution pairs the policy allows but
   flags (word-initial ع-deletion, vowel dropped beside a vowel):
   `data/stimulus/pairs_v2_votes.tsv`, `panel` column per row.
   - chunks c0–c17 (5,400 pairs): Opus x-high, ≥4/5 → 3,559 accepted
   - chunks c18–c22 (1,404 pairs): Sonnet medium, 5/5 → 1,058 accepted
3. **Fused-sentence council** (5 Opus lenses, ≥4/5): 707 of 1,122 accepted,
   `data/stimulus/fusion_v2_votes.tsv`.

Per-judge files: `data/stimulus/council_v2/{fused,pairs}/`.

## Known limitations (sized)

- **Sonnet-judged pairs are lenient.** On 300 pairs judged by both panels,
  Sonnet medium missed 96% of Opus's rejections at 4/5 and 60% at 5/5
  (`experiments/d5_cohere_track/results/council_calibration_pairs_v2_c0_sonnet_medium_vs_opus.md`).
  All 1,404 Sonnet-judged pairs are vowel-beside-vowel, where Opus accepted
  47%; Sonnet at 5/5 accepted 75%, so roughly 390 of its 1,058 accepted
  pairs (~37%) would be Opus rejections. Sonnet-judged pairs appear in 2,795
  emitted rows (0.9%), so about 1,000 rows (~0.35%) likely carry a collision.
- **Fused sentences can be calqued.** The fluency lens alone rejected 343 of
  the 707 accepted fused sentences (English calques like "mashwar 2a2em 3ala");
  the user ruled to keep ≥4/5.
- **Base spellings are machine-chosen** (Command R+ for NileChat, glm-5.3 for
  fused), and they are the normalization targets.
- **Native-speaker sign-off is partial:** the rulings above are signed off,
  the council verdicts themselves are not.

## Regenerate

```bash
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_v2.py
.venv/bin/python experiments/d5_cohere_track/training_data/emit_v2.py
```

Upstream steps (split, collection, fusion, councils, merges) are listed in
`CLAUDE.md` → "v2 fine-tuning dataset". Seeds: generation 13 (per-sentence),
split salt `v2-split-42`, mix seed 42.
