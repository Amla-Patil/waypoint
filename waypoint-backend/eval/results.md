# Evaluation run (backend = tfidf)

Synthetic JDs, demo personas, no human labels: face-validity only.

## Score matrix (rows = persona, cols = job), full matcher

| persona | ml_engineer | data_scientist | devops_engineer | warehouse_ops |
|---|---|---|---|---|
| ml_engineer | 24.0 | 36.0 | 0.0 | 0.0 |
| data_scientist | 16.0 | 40.0 | 7.7 | 0.0 |
| devops_engineer | 0.0 | 0.0 | 15.4 | 0.0 |
| warehouse_ops | 0.0 | 0.0 | 0.0 | 47.1 |

## Config A (keyword baseline) vs full matcher

| metric | A: keyword | full matcher |
|---|---|---|
| top-1 own-role accuracy | 0.75 | 0.75 |
| mean score on own role | 29.7 | 31.6 |
| mean score on other roles | 5.6 | 5.0 |

## Fairness (identity variants: name + gendered words)

- max |score shift| over 16 variants: 0.00; mean 0.00
- selection rate at cutoff 30 per group: A=0.50, B=0.50, C=0.50, D=0.50
- disparate-impact ratio (min/max, four-fifths rule threshold 0.8): 1.00

## Robustness (mean score change vs original, own-role pairs)

- typos: mean +0.0, max +0.0, min +0.0
- shuffle: mean +0.0, max +0.0, min +0.0
- stuffing: mean +32.2, max +42.3, min +26.0
