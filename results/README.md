# V7 result summary

These are **engineering / research rechecks**, not pristine out-of-sample evidence.

| Window | Period | Closed trades | Win rate | Gross P&L | Fees | Net P&L | Net return | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V7 recheck-dev | 2023-01-03 to 2023-01-31 | 2 | 100% | +$72.00 | $4.60 | +$67.40 | +0.067% | 0.90% |
| V7 recheck-check | 2025-06-02 to 2025-06-30 | 6 | 66.7% | +$411.00 | $12.60 | +$398.40 | +0.398% | 0.90% |

## Interpretation

The V7 change removed executable standalone short calls while preserving standalone short puts and allowing a short strangle only when both put and call are independently selected by the joint HJB process.

The January recheck is too small to support a performance claim: only two positions closed. The June recheck has six completed positions, four profitable and two unprofitable. These results support the engineering hypothesis that removing the previously miscalibrated naked-call expression materially reduces the failure mode observed in earlier tests, but they are **not sufficient to establish statistically significant alpha**.

Raw QuantConnect exports are retained in the two result directories so every number can be audited.
