# Calibration under compression

teacher = `teacher` (1543.7M)

## Per-task calibration

| model | params | task | scoring | n | acc | chance | ECE | Brier | gold NLL | AUROC | conf right | conf wrong | raw acc | raw ECE | raw AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher | 1543.7M | mmlu | acc | 14042 | 59.67 | 25.0 | 8.55 | 0.5154 | 0.993 | 0.7798 | 0.7698 | 0.5525 | 59.67 | 8.55 | 0.7798 |
| teacher | 1543.7M | arc_challenge | acc_norm | 1172 | 45.14 | 25.02 | 14.76 | 0.7133 | 1.314 | 0.5464 | 0.3135 | 0.3051 | 41.13 | 38.42 | 0.588 |
| teacher | 1543.7M | arc_easy | acc_norm | 2376 | 71.84 | 25.02 | 38.18 | 0.6308 | 1.164 | 0.6673 | 0.3472 | 0.3098 | 75.42 | 9.67 | 0.725 |
| teacher | 1543.7M | boolq | acc | 3270 | 72.81 | 50.0 | 1.79 | 0.3579 | 0.534 | 0.7006 | 0.758 | 0.6629 | 72.81 | 1.79 | 0.7006 |
| teacher | 1543.7M | piqa | acc_norm | 1838 | 76.12 | 50.0 | 23.03 | 0.4711 | 0.664 | 0.5436 | 0.5313 | 0.5304 | 75.68 | 18.04 | 0.5981 |
| cp17 | 1286.4M | mmlu | acc | 14042 | 36.09 | 25.0 | 4.04 | 0.7198 | 1.338 | 0.6204 | 0.4101 | 0.3696 | 36.09 | 4.04 | 0.6204 |
| cp17 | 1286.4M | arc_challenge | acc_norm | 1172 | 34.04 | 25.02 | 4.23 | 0.7326 | 1.352 | 0.537 | 0.3124 | 0.3065 | 32.85 | 46.15 | 0.5189 |
| cp17 | 1286.4M | arc_easy | acc_norm | 2376 | 61.49 | 25.02 | 28.38 | 0.6555 | 1.208 | 0.6378 | 0.3431 | 0.3118 | 66.58 | 16.2 | 0.6793 |
| cp17 | 1286.4M | boolq | acc | 3270 | 66.36 | 50.0 | 8.51 | 0.4402 | 0.636 | 0.634 | 0.7648 | 0.7039 | 66.36 | 8.51 | 0.634 |
| cp17 | 1286.4M | piqa | acc_norm | 1838 | 71.06 | 50.0 | 18.49 | 0.48 | 0.673 | 0.5235 | 0.5289 | 0.5298 | 70.4 | 22.07 | 0.577 |
| cp29 | 1052.4M | mmlu | acc | 14042 | 22.99 | 25.0 | 32.61 | 0.9099 | 1.668 | 0.5099 | 0.5594 | 0.5549 | 22.99 | 32.61 | 0.5099 |
| cp29 | 1052.4M | arc_challenge | acc_norm | 1172 | 28.75 | 25.02 | 3.95 | 0.7516 | 1.389 | 0.4831 | 0.3093 | 0.315 | 27.56 | 51.34 | 0.4647 |
| cp29 | 1052.4M | arc_easy | acc_norm | 2376 | 53.7 | 25.02 | 20.47 | 0.6842 | 1.261 | 0.5615 | 0.3401 | 0.3245 | 61.24 | 19.1 | 0.6461 |
| cp29 | 1052.4M | boolq | acc | 3270 | 61.28 | 50.0 | 11.28 | 0.5044 | 0.711 | 0.5455 | 0.7326 | 0.7145 | 61.28 | 11.28 | 0.5455 |
| cp29 | 1052.4M | piqa | acc_norm | 1838 | 68.23 | 50.0 | 15.92 | 0.4839 | 0.677 | 0.5294 | 0.5278 | 0.5285 | 68.88 | 23.2 | 0.5465 |

## MMLU answer-position bias (prediction share per option vs gold share)

| model | pred A/B/C/D | gold A/B/C/D | max pred share |
|---|---|---|---|
| teacher | 0.23 / 0.33 / 0.20 / 0.23 | 0.23 / 0.25 / 0.26 / 0.27 | 0.33 |
| cp17 | 0.31 / 0.02 / 0.31 / 0.35 | 0.23 / 0.25 / 0.26 / 0.27 | 0.35 |
| cp29 | 0.98 / 0.01 / 0.01 / 0.01 | 0.23 / 0.25 / 0.26 / 0.27 | 0.98 |

## gsm8k: strict vs flexible exact-match

| model | strict | flexible |
|---|---|---|
| teacher | 61.5 | 61.5 |
| cp17 | 4.0 | 4.0 |
| cp29 | 1.5 | 1.5 |

## Cascade, primary scoring (matches reported acc): student answers its most-confident share, teacher the rest

Cells: combined acc (gain over random escalation). `oracle esc@-1pt` = share escalated to get within 1pt of the teacher, cutoff chosen on the scored labels (upper bound).

| student | task | teacher acc | cov 100% | cov 90% | cov 80% | cov 70% | cov 50% | oracle esc@-1pt |
|---|---|---|---|---|---|---|---|---|
| cp17 | mmlu | 59.7 | 36.1 (+0.0) | 39.0 (+0.6) | 41.6 (+0.8) | 44.0 (+0.8) | 48.6 (+0.8) | 91% |
| cp17 | arc_challenge | 45.1 | 34.0 (+0.0) | 35.8 (+0.6) | 37.8 (+1.5) | 39.4 (+2.0) | 41.7 (+2.1) | 81% |
| cp17 | arc_easy | 71.8 | 61.5 (+0.0) | 63.7 (+1.2) | 65.1 (+1.5) | 66.5 (+1.9) | 68.6 (+2.0) | 79% |
| cp17 | boolq | 72.8 | 66.4 (+0.0) | 68.0 (+1.0) | 69.0 (+1.3) | 70.3 (+2.0) | 71.3 (+1.7) | 62% |
| cp17 | piqa | 76.1 | 71.1 (+0.0) | 72.5 (+1.0) | 73.3 (+1.3) | 73.8 (+1.2) | 74.8 (+1.2) | 58% |
| cp29 | mmlu | 59.7 | 23.0 (+0.0) | 26.7 (+0.1) | 30.8 (+0.4) | 34.4 (+0.4) | 41.4 (+0.1) | 98% |
| cp29 | arc_challenge | 45.1 | 28.8 (+0.0) | 30.2 (-0.2) | 33.0 (+1.0) | 34.9 (+1.2) | 39.3 (+2.4) | 94% |
| cp29 | arc_easy | 71.8 | 53.7 (+0.0) | 56.5 (+1.0) | 58.5 (+1.2) | 60.3 (+1.1) | 63.6 (+0.8) | 91% |
| cp29 | boolq | 72.8 | 61.3 (+0.0) | 62.9 (+0.5) | 63.4 (-0.2) | 65.0 (+0.2) | 66.9 (-0.2) | 83% |
| cp29 | piqa | 76.1 | 68.2 (+0.0) | 70.2 (+1.2) | 72.0 (+2.2) | 73.3 (+2.7) | 74.4 (+2.2) | 58% |

**Holdout cascade (primary scoring (matches reported acc))**: threshold chosen on the calibration half to stay within 1pt of the teacher, applied unchanged to the holdout half.

| student | task | holdout n | escalated | combined acc | teacher acc | student-alone acc |
|---|---|---|---|---|---|---|
| cp17 | mmlu | 6972 | 92% | 59.1 | 59.9 | 35.3 |
| cp17 | arc_challenge | 562 | 82% | 43.6 | 44.5 | 32.7 |
| cp17 | arc_easy | 1181 | 78% | 72.1 | 73.2 | 62.6 |
| cp17 | boolq | 1650 | 54% | 71.3 | 73.1 | 66.4 |
| cp17 | piqa | 894 | 60% | 76.6 | 77.2 | 72.4 |
| cp29 | mmlu | 6972 | 98% | 58.8 | 59.9 | 23.2 |
| cp29 | arc_challenge | 562 | 92% | 43.4 | 44.5 | 27.2 |
| cp29 | arc_easy | 1181 | 91% | 72.1 | 73.2 | 56.0 |
| cp29 | boolq | 1650 | 85% | 72.1 | 73.1 | 61.7 |
| cp29 | piqa | 894 | 75% | 76.8 | 77.2 | 69.1 |

## Cascade, raw-likelihood scoring (true option distribution): student answers its most-confident share, teacher the rest

Cells: combined acc (gain over random escalation). `oracle esc@-1pt` = share escalated to get within 1pt of the teacher, cutoff chosen on the scored labels (upper bound).

| student | task | teacher acc | cov 100% | cov 90% | cov 80% | cov 70% | cov 50% | oracle esc@-1pt |
|---|---|---|---|---|---|---|---|---|
| cp17 | mmlu | 59.7 | 36.1 (+0.0) | 39.0 (+0.6) | 41.6 (+0.8) | 44.0 (+0.8) | 48.6 (+0.8) | 91% |
| cp17 | arc_challenge | 41.1 | 32.8 (+0.0) | 34.4 (+0.7) | 34.9 (+0.4) | 37.0 (+1.7) | 38.1 (+1.1) | 75% |
| cp17 | arc_easy | 75.4 | 66.6 (+0.0) | 69.1 (+1.6) | 70.5 (+2.2) | 72.3 (+3.0) | 74.5 (+3.5) | 50% |
| cp17 | boolq | 72.8 | 66.4 (+0.0) | 68.0 (+1.0) | 69.0 (+1.3) | 70.3 (+2.0) | 71.3 (+1.7) | 62% |
| cp17 | piqa | 75.7 | 70.4 (+0.0) | 72.8 (+1.9) | 73.4 (+2.0) | 73.8 (+1.8) | 74.9 (+1.8) | 42% |
| cp29 | mmlu | 59.7 | 23.0 (+0.0) | 26.7 (+0.1) | 30.8 (+0.4) | 34.4 (+0.4) | 41.4 (+0.1) | 98% |
| cp29 | arc_challenge | 41.1 | 27.6 (+0.0) | 29.6 (+0.7) | 31.6 (+1.3) | 32.3 (+0.7) | 34.6 (+0.3) | 88% |
| cp29 | arc_easy | 75.4 | 61.2 (+0.0) | 64.0 (+1.3) | 66.6 (+2.5) | 68.5 (+3.0) | 71.6 (+3.3) | 79% |
| cp29 | boolq | 72.8 | 61.3 (+0.0) | 62.9 (+0.5) | 63.4 (-0.2) | 65.0 (+0.2) | 66.9 (-0.2) | 83% |
| cp29 | piqa | 75.7 | 68.9 (+0.0) | 70.6 (+1.1) | 72.0 (+1.8) | 72.7 (+1.8) | 74.5 (+2.2) | 60% |

**Holdout cascade (raw-likelihood scoring (true option distribution))**: threshold chosen on the calibration half to stay within 1pt of the teacher, applied unchanged to the holdout half.

| student | task | holdout n | escalated | combined acc | teacher acc | student-alone acc |
|---|---|---|---|---|---|---|
| cp17 | mmlu | 6972 | 92% | 59.1 | 59.9 | 35.3 |
| cp17 | arc_challenge | 562 | 76% | 40.4 | 41.8 | 32.6 |
| cp17 | arc_easy | 1181 | 52% | 75.1 | 76.0 | 67.7 |
| cp17 | boolq | 1650 | 54% | 71.3 | 73.1 | 66.4 |
| cp17 | piqa | 894 | 44% | 76.4 | 77.3 | 71.1 |
| cp29 | mmlu | 6972 | 98% | 58.8 | 59.9 | 23.2 |
| cp29 | arc_challenge | 562 | 85% | 40.7 | 41.8 | 26.7 |
| cp29 | arc_easy | 1181 | 82% | 75.1 | 76.0 | 61.6 |
| cp29 | boolq | 1650 | 85% | 72.1 | 73.1 | 61.7 |
| cp29 | piqa | 894 | 69% | 76.5 | 77.3 | 69.5 |
