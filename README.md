# Volatility Risk Premium Research

A QuantConnect/LEAN research project for testing whether option-implied volatility richness can be harvested more effectively when **signal construction, scenario modelling, portfolio allocation, execution, and exit policy are treated as one system**.

This repository is a research artifact, not a claim of a production-ready trading strategy.

## Research question

The initial thesis was that local volatility-surface dislocations and implied-vs-realized volatility premia could identify option structures with positive expected carry. Early walk-forward tests did not support the first implementation. Rather than reversing the signal or optimizing thresholds until the backtest became positive, the project used controlled ablations to isolate the failure modes.

The main findings that drove the current V7 implementation were:

1. A simple sign inversion did not rescue the strategy under realistic fills.
2. Intraday exit logic was inconsistent with the slower Monte Carlo management assumptions and crystallized many temporary short-volatility mark-to-market losses.
3. Standalone short-call forecasts were materially miscalibrated relative to realized outcomes.
4. Put-side carry behaved substantially better in the diagnostic decompositions.
5. PUT, CALL and CASH therefore need to compete in one portfolio allocation problem rather than in independent optimizers.

## Current V7 architecture

```text
market / option surface
        |
        v
forward realized-vol forecast
        |
        +--> cross-sectional / absolute VRP features
        +--> HMM regime features
        +--> SABR surface features
        |
        v
candidate PUT and CALL legs
        |
        v
Monte Carlo scenario P&L
        |
        v
joint expected-log-wealth / HJB allocation
        |
        +--> PUT selected only  -> standalone short put
        +--> PUT + CALL selected -> short strangle
        +--> CALL selected only -> CASH (research signal retained, no naked call)
        +--> neither            -> CASH
        |
        v
execution + once-per-session package management
```

Legacy SABR/fly candidates remain in the code for research diagnostics but are **research-only in V7** and cannot receive orders.

## Model components

- `forward_vol_model_v2.py` — forward realized-volatility estimates.
- `cross_sectional_vrp_v2.py` — cross-sectional VRP features.
- `hmm_regime_model_v2.py` — market-regime features.
- `sabr_v2.py` / `sabr_*` — volatility-surface diagnostics.
- `strategy_constructor_v2.py` — candidate construction.
- `monte_carlo_multileg_v2.py` — scenario P&L simulation.
- `hjb_position_sizer_v2.py` — expected-log-wealth / HJB-style allocation.
- `execution_v2.py` / `execution_policy_v2.py` — executable pricing, sizing and package management.
- `workflow_v2.py` — end-to-end research workflow.
- `main.py` — QuantConnect algorithm entry point.

## V7 research freeze

The current revision is intentionally narrow:

- joint PUT/CALL/CASH HJB is retained;
- standalone short calls are **not executable**;
- a call is executable only when the matching put is independently selected, forming a strangle;
- standalone puts remain executable with the full package-style risk basis;
- management is once per session to remain consistent with the current MC management frequency;
- IV thresholds, spread filters, MC path count, HJB objective, stop/trailing rules and position-size thresholds were not relaxed to manufacture a positive recheck.

See [`docs/V7_FREEZE.md`](docs/V7_FREEZE.md).

## Current research results

| Window | Closed trades | Win rate | Net P&L | Return | Max DD |
|---|---:|---:|---:|---:|---:|
| Jan-2023 engineering recheck | 2 | 100% | +$67.40 | +0.067% | 0.90% |
| Jun-2025 engineering recheck | 6 | 66.7% | +$398.40 | +0.398% | 0.90% |

The raw QuantConnect exports are under [`results/`](results/).

**These two windows are not presented as out-of-sample validation.** January was used during the debugging process and June was already inspected in V6. The sample is small and does not establish statistical significance. The value of the project is the falsification / ablation process and the traceable evolution from a failing implementation toward a more coherent portfolio expression.

## Selected trade-level observations

In the January V7 recheck, both completed positions were standalone puts. The WMT position finished profitable despite a large adverse mark-to-market excursion, reinforcing the earlier finding that short-volatility carry can be damaged by management rules that realize temporary losses too aggressively.

In June 2025, six standalone put positions closed: four profitable and two unprofitable. Five reached the DTE exit; one exited through the stop mechanism. The aggregate month was positive, but performance was concentrated and the sample remains too small for an alpha claim.

## Reproducibility

The code is designed for QuantConnect/LEAN. To reproduce a test:

1. Create a Python QuantConnect project.
2. Upload the `.py` files in the repository root.
3. Set the desired research window in `qc_config_v2.py`.
4. Keep the model and execution parameters frozen when comparing windows.
5. Export `orders.csv`, `trades.csv` and the backtest JSON after each run.

The historical result files in this repository preserve the exact QuantConnect exports used in the research notes.

## Research limitations

The current evidence is intentionally described conservatively:

- short test windows;
- small V7 trade count;
- earlier iterations influenced model design;
- standalone-call directionality remains inadequately modelled;
- execution/fill assumptions remain backtest approximations;
- historical robustness is not equivalent to prospective live validation.


