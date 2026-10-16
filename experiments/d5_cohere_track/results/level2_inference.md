# Level 2 — can the model tell a real variant from a look-alike?

102 labelled pairs, orders asked: ab, ba.

A model answering SAME to everything scores 1.000 on the SAME column and
0.000 on DIFFERENT, so read the two together. `says-SAME` exposes that bias
directly.

Base rate of SAME in the test set: 0.657. A `says-SAME` rate above that is a bias toward calling things the same word.

| model | acc. on SAME | acc. on DIFFERENT | balanced | says-SAME | order-flip | unparseable |
|---|---|---|---|---|---|---|
| sonct988/gemma4-26b-a4b-it-q4km-256k:latest | 0.978 | 0.529 | **0.753** | 0.804 | 0.078 | 0.000 |
| gemma4:e2b | 0.836 | 0.229 | **0.532** | 0.814 | 0.275 | 0.000 |
| qwen3.8:27b-mlx | 1.000 | 0.014 | **0.507** | 0.995 | 0.010 | 0.000 |
| qwen3.5:0.8b | 1.000 | 0.000 | **0.500** | 1.000 | 0.000 | 0.000 |
