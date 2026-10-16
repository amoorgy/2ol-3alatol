# NileChat word set — stimulus pipeline

Builds the machine-derived probe stimuli for the d5 track: 500 Egyptian
Arabizi word types, each in a real carrier sentence, each rendered as licensed
spelling variants plus (after council review) one trap.

**Provenance rule (hard rule 1).** `nilechat-arabizi-egy` is Command R+
transliteration output — a measurement corpus and probe stimulus, never a
benchmark artifact. Every emitted row carries `provenance`, and any headline
must also be shown on the human-authored arm (`data/benchmark/pairs.csv`).

## Why variants are GENERATED and traps are MINED

Measured on this corpus: 93% of co-skeleton pairs converge under
`classify_edge`, and that bucket mixes true variants (`men`/`min`) with
different words (`men`/`meen` = من/مين) indistinguishably — orthographic
explainability is not word identity. So:

- The SAME arm cannot be mined. Respelling a known base word with licensed
  rules makes same-word identity true **by construction**
  (`src/franco_eval/licensing.py`).
- The DIFFERENT arm is exactly those converged look-alikes, ranked by PPMI
  context dissimilarity (`src/franco_eval/context.py`, calibrated on the
  human arm at AUC 0.873) and adjudicated by a five-lens council.

## Pipeline

```bash
# 1. frequency table + variance diagnostic + 500-word stratified selection
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/build_wordset.py

# 2. licensed variants -> committed stimulus file (re-run after council votes)
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/build_variants.py

# 3. trap candidates ranked for the council
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/mine_traps.py
```

Nothing is stochastic; there is no seed to record. Smoke-run any script with
`--limit-docs 20000` (~5s).

## Files

| path | tracked | role |
|---|---|---|
| `data/generated/nilechat_wordfreq.tsv` | no | corpus frequency table |
| `data/generated/nilechat_candidates.json` | no | 500 words + carriers |
| `data/generated/nilechat_trap_candidates.tsv` | no | ranked trap shortlist |
| `data/stimulus/nilechat/stimuli_v1.csv` | **yes** | the probe stimuli |
| `data/stimulus/nilechat/{rule,variant,trap}_votes.tsv` | **yes** | council verdicts |
| `results/wordset_diagnostic.{json,md}` | yes | variance + hygiene report |
| `results/context_calibration.json` | yes | PPMI threshold, human-arm AUC |

Council-invalidated rows are **flagged (`council_valid=false`), never
deleted** — a rule that is valid in general but wrong for one word
(`men`→`meen`) is the most diagnostic cell in the experiment.
