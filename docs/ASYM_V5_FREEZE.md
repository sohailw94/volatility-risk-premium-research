# ASYM_V5 research freeze

Purpose: port the WF_A ablation finding into the full SABR/VRP/HMM/MC/HJB project without changing the primary capped local-fly thesis.

## Frozen research rule

1. PRIMARY branch is unchanged: local SABR residual + VRP -> capped fly -> MC -> HJB -> execution.
2. If PRIMARY submits no order, absolute carry generates an OTM ~16-delta PUT candidate and CALL candidate separately.
3. Each side must itself have IV/RV >= `STRANGLE_MIN_IV_RV` and positive variance premium.
4. PUT and CALL are run through separate Monte Carlo jobs and separate HJB optimizations against the same pre-order portfolio state.
5. Post-HJB expression:
   - PUT approved, CALL rejected -> `SHORT_PUT_VRP`.
   - CALL approved, PUT rejected -> `SHORT_CALL_VRP`.
   - BOTH approved -> `SHORT_STRANGLE_APPROVED`.
   - neither -> cash.
6. A strangle is never allowed to make an otherwise rejected leg tradable.
7. Standalone PUT retains the full pair-style management risk proxy; removing the call does not halve the put's exit denominator.
8. CALL capital is frozen at 50% of PUT capital:
   - PUT 7.50% position/single/ticker cap, max 2 contracts.
   - CALL 3.75% position/single/ticker cap, max 1 contract.
   - paired strangle is call-limited at 3.75%, max 1 package.
9. ASYM carry positions use package-only management: package stop/trail + DTE exit. The 15% single-leg stop/trail remains active only for non-ASYM structures such as primary capped flies.
10. The 7-DTE and 1-DTE SPY/NDXP fallback keeps the capped index-fly attempt, then uses the same independent-leg ASYM rule if the fly submits nothing.

## Validation discipline

Run `WF_A`, `WF_B`, `WF_C`, `WF_D` with the exact same source and parameters. The only permitted difference between the four run packages is `RUN_WINDOW` in `qc_config_v2.py`.

Do not tune thresholds after viewing any one window. Record separately:
- primary fly submissions;
- standalone put submissions;
- standalone call submissions;
- both-approved strangles;
- MC expected PnL / POP / CVaR by role;
- HJB selection rates by role;
- fill/rejection rates;
- realized PnL by expression;
- return, drawdown, Sharpe, profit factor, and fees.

This is a post-ablation development variant, not an untouched holdout result.
