# Showcase: retrieval robustness under Franco spelling variation

120 fragment queries (4-word windows) x 3 variants per intensity; corpus of 489 human-authored seed messages; mean over seeds (13, 17, 29, 41, 53).

## Synthetic perturbation (taxonomy-generated variants)

`norm=source` is the leakage diagnostic: the fraction of variants that normalize to the exact normalized source — an oracle head-start the normalized arm gets because normalize.py inverts part of taxonomy.py by construction. Read the normalized column with that discount; the attested table below is free of this artifact.

| intensity | recall@1 raw | recall@1 normalized | C@3 raw | C@3 normalized | norm=source |
|---|---|---|---|---|---|
| 0.0 (clean) | 0.933 | 0.938 | — | — | — |
| 0.2 | 0.895 | 0.915 | 0.860 | 0.910 | 43% |
| 0.4 | 0.888 | 0.914 | 0.833 | 0.902 | 36% |
| 0.6 | 0.863 | 0.902 | 0.788 | 0.878 | 26% |
| 0.8 | 0.833 | 0.888 | 0.737 | 0.865 | 18% |

## Attested variation (real spelling swaps from mined clusters)

Queries where words are replaced by *attested* alternate spellings (mined from the corpus itself), not synthetic rules — the honest test.

| system | recall@1 | n queries (mean/seed) |
|---|---|---|
| raw | 0.850 | 58 |
| normalized | 0.871 | 58 |
