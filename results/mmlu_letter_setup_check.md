# MMLU answer-letter check under a different setup

Exploratory, 2026-11-05. Five-shot subject prompts on 384 unique MMLU **validation** items (three 128-item development splits);
option-likelihood scoring, one request at a time. Dataset revision `c30699e8356da336a370243923dbaf21066bb9fe`, teacher revision
`8faed761d45a263340a0528343f099c05c9a4323`, request manifest SHA256 `bddcc6d92e189bc8465322edcfacdbd5ab8af93c439b3f695cd9036bac000876`.
It is not a reproduction of the main MMLU evaluation, and no weights were edited.

| model | accuracy | predicted A | B | C | D |
|---|---:|---:|---:|---:|---:|
| gold answers | - | 27.1% | 23.2% | 26.8% | 22.9% |
| teacher | 62.5% | 22.7% | 28.9% | 27.9% | 20.6% |
| cp17 | 40.4% | 48.2% | 6.2% | 15.9% | 29.7% |
| cp29 | 24.7% | 20.1% | 4.7% | 2.3% | 72.9% |

cp29's favored letter is D on each split (75.8%, 68.8%, 74.2%), where the main evaluation has A on 98%. Prompt and split both
differ, so this does not show which of them moved it. Raw per-item records are not included.
