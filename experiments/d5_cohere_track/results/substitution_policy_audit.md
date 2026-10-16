# v2 substitution policy vs. committed a_v1 verdicts

5,400 fully judged a_v1 permutation rows; baseline error with no policy 17.9%.

| policy | catch rate | false block | residual error | rows passing |
|---|---|---|---|---|
| rules_only_out_of_sample | 73.3% | 9.9% | 6.1% | 4,253 |
| full_policy_in_sample | 97.9% | 19.7% | 0.6% | 3,578 |

`rules_only` is the honest estimate for unseen substitution pairs; `full` is in-sample (its bad_pairs come from these votes).
