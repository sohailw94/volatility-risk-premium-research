# BOLD_V4 audit and change log

## Why the previous build was sparse

The active path stacked several gates before an order could exist:

1. Primary SHORT_RICH required all of:
   - local SABR residual >= 2.0 vol points,
   - robust residual z >= 1.50,
   - IV / forecast RV >= 1.05,
   - positive variance premium.
2. The forecast RV already included continuous variance + historical jump variance + term/event variance, which made the short-vol gate harder.
3. Even after a signal, the only trade expression was a three-leg local fly.
4. Every fly leg needed a positive bid and had to pass a relative bid/ask spread cap. Cheap wings can fail a relative spread test even when the absolute spread is only one or two cents.
5. Credit structures additionally needed executable natural credit >= 80% of midpoint credit and >= $0.05.
6. HJB then applied 5% single/ticker and 15% total defined-risk limits.
7. The production MC still used the legacy 50%-premium profit target and 2x-premium stop, which did not match the requested stepped management.

The observed WF_A result is consistent with this: only a WMT BUY_CHEAP_LOCAL_FLY survived to a trade. That is a long-cheap-vol expression, not evidence that the short-rich branch was broadly active.

## BOLD_V4 entry changes

Primary local-fly thesis:
- MIN_SURFACE_RESIDUAL: 0.020 -> 0.012
- MIN_SURFACE_Z: 1.50 -> 1.00
- MIN_SHORT_IV_RV: 1.05 -> 1.02
- MAX_LONG_IV_RV: 0.95 -> 0.98

Liquidity:
- Zero bids are still rejected.
- Relative spread caps remain.
- A high relative spread is allowed when the absolute spread is <= $0.03. This prevents penny wings such as $0.01/$0.02 from being rejected solely because the percentage spread is large.
- MIN_CREDIT_FRACTION_OF_MID: 0.80 -> 0.65
- MIN_NET_CREDIT: $0.05 -> $0.03

Short-strangle fallback:
- New STRANGLE_FALLBACK branch after the primary fly branch.
- Uses 16-delta target shorts, with an 8-25 delta admissible band.
- Requires selected call/put average IV >= continuous realized-vol forecast (STRANGLE_MIN_IV_RV = 1.00) and positive continuous variance premium.
- The production MC still includes jump/stress dynamics.
- At most 12 strangle candidates are sent to MC/HJB per research cycle.
- Short-dated SPY/NDXP fallback compares both capped flies and short strangles.

## Risk allocation changes

Capped structures:
- position percentage per full HJB action: 10%
- max single-trade risk: 10%
- max ticker risk: 10%

Portfolio:
- max total risk: 25%
- max new trades/day: 5

Uncapped structures:
- explicit risk_class = UNCAPPED
- margin-style sizing proxy = 12% of underlying notional per contract
- max single/ticker risk = 7.5%
- max 1 contract per uncapped trade
- HJB actions for uncapped structures are binary 0 or 1, avoiding fractional-contract sizing fiction.

The uncapped max_loss field is a sizing proxy, not a claim of bounded loss. true_max_loss is explicitly stored as infinity.

## Live management changes

Management runs every 5 minutes while the market is open.

Leg rules:
- at -15% leg P&L: close the leg if doing so reduces risk;
- if closing the leg would remove a protective long wing and create more tail risk, escalate to a full-structure exit;
- at +25% leg P&L: arm a +25% hard floor and set the next rung to +35%;
- at +35%, move the floor to +35% and the next rung to +45%, etc. in 10-point steps.

Whole-strategy rules:
- risk basis = actual structure risk_dollars (or gross premium basis if larger);
- -8% of risk basis triggers the normal limit-exit ladder;
- if deterioration reaches -10% while still open, force the remaining holdings out;
- initial profit-trail threshold is the larger dollar amount of:
  - 15% of strategy risk basis, or
  - 25% of the largest leg's entry premium;
- after the first profit rung is reached, the next target advances by 10% of strategy risk basis and the prior rung becomes the hard floor.

MC/HJB:
- production MC was changed to use the same 8% strategy stop and stepped profit/leg rules.
- MC treats a leg-level stop/trail as a full-structure exit, which is conservative relative to the live engine's ability to close only a risk-reducing short leg.

## Diagnostics

New selected-trade log:
V3_PICK|<mode>|T=<ticker>|S=<structure>|EV=<...>|POP=<...>|A=<...>|RISK=<CAPPED/UNCAPPED>

Decision log still reports:
P = primary local fly
S = broad short-strangle fallback
F7 = 7-DTE SPY/NDXP
F1 = 1-DTE SPY/NDXP

## Regression checks performed

- All 23 Python files compile.
- Active workflow imports successfully under a QC API stub.
- execution_v2.py is under QuantConnect's 32,000-character limit.
- Synthetic case with tiny SABR residual but IV > continuous RV:
  - primary local signal = 0
  - strangle signal = 1
  - executable short strangle = 1
- HJB selects the uncapped strangle under positive scenario EV and returns cash under negative scenario EV.
- A $0.01/$0.02 wing passes the hybrid spread rule.
- A zero-bid leg is still rejected.
- Leg trail test:
  - +26% arms +25% floor / +35% next target
  - +36% moves floor to +35% / next target to +45%
  - retrace to +34% triggers the hard profit exit.

## Research-label warning

This is no longer the frozen original thesis. It is a deliberate BOLD_V4 development variant requested after observing sparse trading. Keep the old frozen results separately. Do not present BOLD_V4 validation as untouched out-of-sample evidence for the original architecture.
