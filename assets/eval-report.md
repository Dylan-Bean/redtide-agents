# Evaluation report

## Config: baseline

- tasks 30, passed **28 (93%)**, avg score 0.92
- avg latency 216s/task, total tokens 644848

| category | passed/total |
|---|---|
| 发生机制 | 10/10 |
| 主要藻种 | 4/6 |
| 监测预警 | 7/7 |
| 治理修复 | 5/5 |
| 预警判定 | 2/2 |

Failure attribution: no_report×2

## Config: optimized

- tasks 30, passed **28 (93%)**, avg score 0.93
- avg latency 289s/task, total tokens 848617

| category | passed/total |
|---|---|
| 发生机制 | 8/10 |
| 主要藻种 | 6/6 |
| 监测预警 | 7/7 |
| 治理修复 | 5/5 |
| 预警判定 | 2/2 |

Failure attribution: evidence_underused×1, no_report×1

## Optimization result

pass rate 93% → **93%**
