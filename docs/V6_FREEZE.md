# V6 frozen diagnostic revision

Changes from ASYM_V5 are limited to three audit-driven fixes:

1. **Model/live management cadence aligned**: both MC and live/backtest management evaluate exits once per session. The old 5-minute live manager was removed; the existing MC daily management logic is unchanged. Same-day management after a 10:00 entry is skipped.
2. **Joint PUT/CALL/CASH HJB**: all executable asymmetric carry legs share one MC scenario matrix and one expected-log-wealth optimization. A call no longer receives its own independent portfolio budget.
3. **Carry before legacy fly**: asymmetric carry is evaluated first. The local-SABR fly branch is only attempted when carry submits nothing. The same ordering is used for 7-DTE/1-DTE index fallback.

No IV/RV threshold, SABR threshold, liquidity threshold, call/put capital budget, stop percentage, trail percentage, or slippage assumption was changed.

Speed windows were pre-registered before running V6:
- FAST_DEV: 2023-01-03 to 2023-01-31 (first calendar month of WF_A)
- FAST_CHECK: 2025-06-02 to 2025-06-30 (first calendar month of historical holdout)

They are diagnostic/development windows, not untouched OOS tests. After the rule is frozen, rerun WF_A/B/C/D without changing code.
