# d5 — Cohere track: does anything in the pipeline know two spellings are one word?

One question, asked at two levels.

> Given two Egyptian Arabizi spellings, are they **the same word** or **different words**?

|  | Section SAME (67 pairs) | Section DIFFERENT (35 pairs) |
|---|---|---|
| **Level 1 — tokenizer** | how much do the two spellings share? | same measurement |
| **Level 2 — inference** | does the model answer SAME? | does it answer DIFFERENT? |

Section DIFFERENT exists so a model cannot win by always answering SAME. Read the
two columns together or the numbers mean nothing.

## Run it

```bash
# 1. rebuild the test set from its three inputs
python3 experiments/d5_cohere_track/e1_variant_probe/build_pairs.py

# 2. Level 1 — tokenizers only, no model, ~1 min
.venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level1_tokenizer.py

# 3. Level 2 — needs `ollama serve`; responses are cached so reruns are free
.venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level2_inference.py \
    --models "gemma4:e2b,qwen3.8:27b-mlx" --both-orders
```

## Where the test set comes from

```
data/benchmark/probe_hand.json         11 clusters written by the author (has Arabic + meaning)
data/benchmark/variant_clusters.tsv    57 clusters mined from real Franco text (no Arabic)
data/benchmark/probe_annotations.tsv   meaning partitions, e.g.  3ala 3la|3aila
                  │                    pairs inside a partition = SAME
                  │                    pairs crossing partitions = DIFFERENT
                  ▼  build_pairs.py
data/benchmark/pairs.csv               102 labelled pairs — the only test set
```

To change a label, edit `probe_annotations.tsv` and re-run `build_pairs.py`.
Clusters not listed there default to "all forms are the same word".

## What each data directory is for

| dir | needed by | in git? |
|---|---|---|
| `data/benchmark/` | **this track** — the three inputs and `pairs.csv` | yes |
| `data/seed/` | D1 showcase; the only human-authored Franco text we have | yes, 51 KB |
| `data/generated/` | Level 2 response cache, created on demand | no |

This track reads only tracked files, so it reproduces from a fresh clone.
`variant_clusters.tsv` is machine-mined from `data/seed/` by
`experiments/d1_benchmark/run_showcase.py`, which is why it has no Arabic column.
`data/interim/` is now empty: it held a 492-row re-export of the seed files that
the showcase's provenance filter discarded in full, so it contributed nothing to
any result. The directory and the `INTERIM` code path stay as the intake point
for a real collected corpus.

## The two metrics that matter

**Level 1 — separability AUC.** The probability that a DIFFERENT pair fragments
more than a SAME pair. `0.5` means fragmentation carries no information, so the
tokenizer gives the model no help. Every tokenizer measured lands at 0.56–0.63.

**Level 2 — accuracy on each section, plus `says-SAME`.** The base rate of SAME in
the test set is 0.657; a model answering SAME more often than that is biased toward
calling things the same word. `order-flip` is how often swapping the two words
changes the verdict — but a low flip rate for an always-SAME model is degeneracy,
not stability, so read it next to `says-SAME`.

## Code

| file | role |
|---|---|
| `src/franco_eval/phenomena.py` | digit inventory, rule licensing, `classify_edge`. stdlib only |
| `src/franco_eval/tokenizer_metrics.py` | `frag` (1 − Jaccard over token ids), `fertility`, `auc`. stdlib only |
| `e1_variant_probe/build_pairs.py` | builds `pairs.csv` |
| `e1_variant_probe/run_level1_tokenizer.py` | Level 1 |
| `e1_variant_probe/run_level2_inference.py` | Level 2 |

Both `src/franco_eval` modules take plain functions rather than model objects, so
`tests/test_phenomena.py` and `tests/test_tokenizer_metrics.py` run with no
downloads and no venv.

## Open, and blocking good claims

1. `pairs.csv` has an empty `confirmed` column. Every DIFFERENT label is Claude's
   reading of Arabic, not a native speaker's. `fel`/`fil` and `bro`/`bara` are
   explicitly uncertain.
2. `gender` and `number` are 3 pairs each, and they carry the most interesting
   Level 2 result (models collapse singular/plural and masculine/feminine). Too
   thin to defend.
3. No Cohere model at Level 2 — `ollama pull aya-expanse:8b`. Its tokenizer is
   already in Level 1, so pulling it connects the two levels for one model.

---

**2026-08-22 update.** The track now has a third level (in-context
word-meaning consistency) and a NileChat-derived stimulus arm. Pipeline,
provenance rules and current results: [`nilechat_wordset/README.md`](nilechat_wordset/README.md)
and [`results/README.md`](results/README.md). The two-level design above is
retained verbatim; Level 2 is frozen with its committed results.
