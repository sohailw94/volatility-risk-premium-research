# Volatility Risk Premium Research

A QuantConnect/LEAN research project on **when option-implied volatility is rich enough to sell, how to size that exposure, and how execution/exit rules change the realized edge**.

The project combines forward-vol forecasting, cross-sectional VRP, HMM regime features, SABR surface diagnostics, Monte Carlo scenario P&L, and expected-log-wealth / HJB-style portfolio allocation.

## Research question

Can volatility risk premium be harvested more selectively than simply selling straddles or selling IV whenever it exceeds realized volatility?

The original implementation **lost money**. I kept the failed versions and used controlled ablations to identify why rather than tuning thresholds until the backtest turned positive.

## What the audit found

- Reversing the original signal did not rescue performance under realistic fills.
- Live exits were being checked far more frequently than the Monte Carlo model assumed, crystallizing temporary short-volatility losses.
- Standalone short-call forecasts were badly miscalibrated relative to realized outcomes.
- Put-side carry was materially stronger in the diagnostic decompositions.
- PUT, CALL and CASH therefore needed to compete in one portfolio allocation problem.

## Current V7 pipeline

```text
option surface + price history
        ↓
forward realized-vol forecast
        ↓
VRP + HMM regime + SABR features
        ↓
PUT / CALL candidates
        ↓
Monte Carlo scenario P&L
        ↓
joint expected-log-wealth / HJB allocation
        ↓
PUT only      → short put
PUT + CALL    → short strangle
CALL only     → cash
        ↓
executable pricing + once-per-session risk management
```

Standalone calls remain visible to the research process but are **not executable in V7** unless the corresponding put is also independently selected. Legacy SABR/fly candidates are retained for diagnostics but are research-only.

## Current V7 rechecks

| Window | Trades | Win rate | Net P&L | Return | Max DD |
|---|---:|---:|---:|---:|---:|
| Jan-2023 | 2 | 100% | +$67.40 | +0.067% | 0.90% |
| Jun-2025 | 6 | 66.7% | +$398.40 | +0.398% | 0.90% |

These are **engineering rechecks, not out-of-sample validation**. The sample is small and does not establish statistical significance.

The important result is the research process: the original strategy failed, the failure was decomposed, model/execution mismatches were isolated, and V7 expresses only the exposures that survived those diagnostics.

## Repository map

- `main.py` — QuantConnect entry point
- `workflow_v2.py` — end-to-end research pipeline
- `forward_vol_model_v2.py` — forward realized-vol model
- `cross_sectional_vrp_v2.py` — volatility-risk-premium features
- `hmm_regime_model_v2.py` — regime model
- `sabr_v2.py` / `sabr_*` — volatility-surface diagnostics
- `monte_carlo_multileg_v2.py` — scenario P&L engine
- `hjb_position_sizer_v2.py` — portfolio allocation
- `execution_v2.py` / `execution_policy_v2.py` — execution and risk management
- `docs/RESEARCH_CHRONOLOGY.md` — how the thesis evolved
- `results/` — published order/trade exports and result summary

## Reproduce

1. Create a Python project in QuantConnect/LEAN.
2. Upload the root `.py` files.
3. Select a frozen test window in `qc_config_v2.py` and run without changing the research parameters.

Full QuantConnect engine JSON is kept outside the public repo; the public `results/` directory contains the compact order/trade exports used to audit the headline figures.

## Limitations

This remains research, not a production trading system. Current evidence is limited by small V7 samples, historical iteration on earlier windows, backtest fill assumptions, and unresolved directional modelling for standalone short calls.
