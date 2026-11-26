# Six-model fixed-test analysis

Offline rescoring of saved predictions; 200 questions per dataset, K=4.

| Dataset | Model | Correct /200 | Strict / reference slots | Compatible / reference slots |
|---|---|---:|---:|---:|
| gsm8k | v1 | 7 | 4/634 | 8/634 |
| gsm8k | prefix_mask | 4 | 3/634 | 5/634 |
| gsm8k | p1 | 3 | 5/634 | 9/634 |
| gsm8k | p2 | 4 | 5/634 | 7/634 |
| gsm8k | p3 | 3 | 2/634 | 4/634 |
| gsm8k | p4 | 6 | 3/634 | 9/634 |
| gsm_hard | v1 | 1 | N/A | N/A |
| gsm_hard | prefix_mask | 1 | N/A | N/A |
| gsm_hard | p1 | 1 | N/A | N/A |
| gsm_hard | p2 | 1 | N/A | N/A |
| gsm_hard | p3 | 2 | N/A | N/A |
| gsm_hard | p4 | 2 | N/A | N/A |
| svamp | v1 | 2 | N/A | N/A |
| svamp | prefix_mask | 2 | N/A | N/A |
| svamp | p1 | 2 | N/A | N/A |
| svamp | p2 | 1 | N/A | N/A |
| svamp | p3 | 5 | N/A | N/A |
| svamp | p4 | 1 | N/A | N/A |

## Paired answer changes versus v1

| Dataset | Model | Gained | Lost | Net correct |
|---|---|---:|---:|---:|
| gsm8k | prefix_mask | 1 | 4 | -3 |
| gsm8k | p1 | 1 | 5 | -4 |
| gsm8k | p2 | 1 | 4 | -3 |
| gsm8k | p3 | 2 | 6 | -4 |
| gsm8k | p4 | 2 | 3 | -1 |
| gsm_hard | prefix_mask | 1 | 1 | +0 |
| gsm_hard | p1 | 0 | 0 | +0 |
| gsm_hard | p2 | 1 | 1 | +0 |
| gsm_hard | p3 | 1 | 0 | +1 |
| gsm_hard | p4 | 1 | 0 | +1 |
| svamp | prefix_mask | 2 | 2 | +0 |
| svamp | p1 | 0 | 0 | +0 |
| svamp | p2 | 1 | 2 | -1 |
| svamp | p3 | 5 | 2 | +3 |
| svamp | p4 | 1 | 2 | -1 |

## Problem groups

Full answer and explanation changes, including unchanged and degraded groups, are in problem_types.csv.
Each axis is evaluated separately. Counts below overlap across axes; they are not counts of unique questions.

| Dataset | Model | EM improved / unchanged / degraded groups |
|---|---|---|
| gsm8k | prefix_mask | 1 / 6 / 7 |
| gsm8k | p1 | 0 / 4 / 10 |
| gsm8k | p2 | 1 / 6 / 7 |
| gsm8k | p3 | 2 / 4 / 8 |
| gsm8k | p4 | 1 / 8 / 5 |
| gsm_hard | prefix_mask | 1 / 4 / 1 |
| gsm_hard | p1 | 0 / 6 / 0 |
| gsm_hard | p2 | 1 / 4 / 1 |
| gsm_hard | p3 | 2 / 4 / 0 |
| gsm_hard | p4 | 2 / 4 / 0 |
| svamp | prefix_mask | 0 / 11 / 0 |
| svamp | p1 | 0 / 11 / 0 |
| svamp | p2 | 0 / 7 / 4 |
| svamp | p3 | 5 / 6 / 0 |
| svamp | p4 | 2 / 5 / 4 |

## Scope and limitations

- Historical latency only: no complete six-model same-session benchmark and no cross-session speedup claim.
- GSM8K uses original strict and conservative step alignment; SVAMP has whole-expression compatibility only.
- GSM-Hard has no reference chain: reference-aligned metrics are N/A, not zero.
- K=4; five decoded slots include a final diagnostic slot not consumed by the reasoner.
- Overlapping grouping axes must not be added together; small groups are flagged.
- Single seed; v1 selection previously used test results; exploratory comparisons only.
- Decoded explanation matches and arithmetic validity do not establish causal latent faithfulness.
- Weights are not loaded or rehashed here; checkpoint identities use the previous verified backup audit.

summary.csv contains answer/reference lengths and mean, median and P95 historical latency.
steps.csv contains all 18,000 decoded slot scores; paired_questions.csv covers all 3,000 model-versus-v1 question pairs.
