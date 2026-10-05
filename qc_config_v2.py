START_CASH = 100_000.0

# ------------------------------------------------------------
# VALIDATION WINDOWS -- evaluation periods DO NOT OVERLAP.
# The 2024-05-01 -> 2025-05-30 interval is deliberately quarantined because
# the earlier V2 project was already examined there.  2026-01-02 -> 2026-02-13
# is DEVELOPMENT because it was used for the thesis audit.
# ------------------------------------------------------------
RUN_WINDOW = "V7_RECHECK_DEV"  # V7_RECHECK_DEV / V7_RECHECK_CHECK / V7_ROBUSTNESS / WF_A / WF_B / WF_C / WF_D / HIST_HOLDOUT

WINDOWS = {
    # V7 engineering rechecks. Already seen; never call these OOS.
    "V7_RECHECK_DEV": ((2023, 1, 3), (2023, 1, 31)),
    "V7_RECHECK_CHECK": ((2025, 6, 2), (2025, 6, 30)),
    # Pre-registered after the V6 call-side failure diagnosis. This is a fast
    # historical robustness check, not a pristine prospective OOS claim.
    "V7_ROBUSTNESS": ((2026, 7, 1), (2026, 7, 31)),
    # V6 pre-registered speed windows.  Chosen chronologically, NOT because of
    # profitability: first calendar month of WF_A and first calendar month of
    # the historical holdout.  These are development diagnostics, not OOS proof.
    "FAST_DEV": ((2023, 1, 3), (2023, 1, 31)),
    "FAST_CHECK": ((2025, 6, 2), (2025, 6, 30)),
    # Already-seen engineering check: use ONLY to verify the fixes below.
    "PLUMBING_CHECK": ((2023, 3, 13), (2023, 3, 24)),
    # Historical post-fix walk-forward/revalidation.  This period was exposed to
    # the older V2 research, so it is NOT the final untouched holdout.
    "POSTFIX_WF": ((2024, 5, 1), (2025, 5, 30)),
    "WF_A": ((2023, 1, 3), (2023, 4, 28)),
    "WF_B": ((2023, 5, 1), (2023, 8, 31)),
    "WF_C": ((2023, 9, 1), (2023, 12, 29)),
    "WF_D": ((2024, 1, 2), (2024, 4, 30)),
    "HIST_HOLDOUT": ((2025, 6, 2), (2025, 12, 31)),
    # Strict prospective OOS after the 2026-09-24 clean-mechanics freeze. Do not run/score
    # until the period has actually elapsed.
    "PROSPECTIVE_OOS": ((2026, 9, 25), (2026, 12, 31)),
}
BACKTEST_START, BACKTEST_END = WINDOWS[RUN_WINDOW]
TRADE_START = BACKTEST_START

TICKERS = [
    'AAPL','AMD','AMZN','SPY','QQQ','IWM','UVIX','SOXX','NVDA','INTC','META','NFLX','V','RGTI','TSLA','MU',
    'AMAT','MRVL','DELL','PLTR','GLD','GDX','SLV','SIL','COPX','FCX','SCCO','DBB','XLE','XOM','CVX','COP',
    'XLP','WMT','COST','KO','PEP','CAR','HTZ'
]
ANCHOR = 'SPY'

# Point-in-time price history.  Bulk history is used, so retaining 420 bars is
# no longer the major runtime bottleneck and preserves the HMM history depth.
PRICE_LOOKBACK_BARS = 420
PRICE_LOOKBACK_DAYS = 420
MIN_HISTORY_ROWS = 80
TRADING_DAYS = 252.0

# -------- THESIS V3: event/jump-aware forward realized variance --------
FORWARD_HORIZON_DAYS = 20
EWMA_LAMBDA = 0.94
OU_VARIANCE_HALFLIFE = 20.0
FORWARD_OU_WEIGHT = 0.60
FORWARD_EWMA_WEIGHT = 0.40
JUMP_Z = 2.50
JUMP_LOOKBACK = 120

# -------- THESIS V3: target option surface --------
TARGET_DTE = 20
OPTION_MIN_DTE = 10
OPTION_MAX_DTE = 35
OPTION_STRIKE_RANGE = 15
MAX_OPTION_CONTRACTS_PER_TICKER = None

# Mechanical liquidity gate.  Rich IV is not tradable alpha if the chain has
# zero bids or a very wide ask.  These are execution/data-quality rules, not
# performance-tuned alpha thresholds.
PRIMARY_MAX_RELATIVE_BID_ASK_SPREAD = 0.50
LIQUIDITY_ABSOLUTE_SPREAD_OVERRIDE = 0.03
MAX_DEBIT_MULTIPLE_OF_MID = 1.25

# If the normal universe produces no executable strategy, check only the most
# liquid short-dated index products.  7 DTE is preferred; 1 DTE is the final
# fallback.  The same alpha/MC/HJB stack still has to approve the trade.
ENABLE_LIQUID_FALLBACK = True
FALLBACK_TICKERS = ('SPY', 'NDX')
FALLBACK_DTES = (7, 1)
FALLBACK_MIN_DTE = 1
FALLBACK_MAX_DTE = 7
FALLBACK_MAX_RELATIVE_BID_ASK_SPREAD = 0.30
# Fallback trades are allowed to express the short-dated thesis instead of being
# closed immediately by the primary 7-DTE exit rule.  Both MC and live
# management read this per-trade value.
FALLBACK_EXIT_DTE = 0
NDX_OPTION_TARGET = 'NDXP'

# Frozen from the successful 2026-01-02 -> 2026-02-13 development audit.
DEFAULT_SABR_BETA = 0.60
SABR_MIN_POINTS = 7
SABR_MIN_IV = 0.03
SABR_MAX_IV = 3.50
SABR_MAX_ACCEPTED_RMSE = 0.10
SABR_MAX_NFEV = 150
MIN_SURFACE_RESIDUAL = 0.012     # relaxed from 2.0 to 1.2 vol points
MIN_SURFACE_Z = 1.00
MIN_SHORT_IV_RV = 1.02
MAX_LONG_IV_RV = 0.98

# Both branches are predeclared and frozen.  Report them separately as well as
# combined; do not switch one off after seeing a validation block.
ENABLE_SHORT_RICH = True
ENABLE_LONG_CHEAP = True

# Existing HMM / Monte Carlo / allocator settings.  Keep the same values for
# every validation window.
HMM_MIN_OBSERVATIONS = 80
HMM_N_STATES = 3
HMM_RANDOM_STATE = 42

# 5k is deliberately frozen for the validation program to keep runtime bounded.
# Do not increase only for a favorable/unfavorable block.
N_PATHS = 5_000
MC_MAX_DAYS = 60
RISK_FREE_RATE = 0.04
VOL_MULTIPLIERS = (0.80, 1.00, 1.20)
PROFIT_TARGET_FRACTION = 0.50  # legacy; live BOLD_V4 management uses stepped trailing rules below
STOP_LOSS_MULTIPLE = 2.00       # legacy; live BOLD_V4 management uses stepped stop rules below
MIN_EXIT_DTE = 7
CONTRACT_MULTIPLIER = 100.0

MIN_DOLLAR_COST_PER_LEG = 0.50
BASE_COMPLEX_ORDER_CAPTURE = 0.35
SIZE_IMACT_PER_EXTRA_CONTRACT = 0.05

ACTION_GRID = (0.00, 0.25, 0.50, 1.00)
POSITION_PCT_PER_ACTION = 10.0
MAX_TOTAL_DEFINED_RISK_PCT = 25.0
MAX_SINGLE_TRADE_RISK_PCT = 10.0
MAX_TICKER_RISK_PCT = 10.0
HJB_RANDOM_SEED = 42
HJB_RANDOM_STARTS = 24
HJB_MAX_PASSES = 25

# Execution plumbing -- retain your current execution_v2.py.
EXECUTE_ORDERS = True
ENTRY_ORDER_TIMEOUT_MINUTES = 10
ENTRY_EXECUTION_MODE = 'NATURAL_MARKETABLE'
# Limit tolerance only; HJB/MC uses executable natural, not natural+cushion.
ENTRY_MARKETABLE_CUSHION = 0.01
MIN_CREDIT_FRACTION_OF_MID = 0.65
MIN_NET_CREDIT = 0.03

# -------- ASYM_V5: independently approved short-vol legs --------
# The primary capped SABR-local-fly branch is unchanged.  If it submits nothing,
# broader absolute carry is expressed as independently researched OTM put and call
# candidates.  Each side gets its OWN MC + HJB decision.  A strangle is created only
# when both sides independently receive a positive HJB action.
ENABLE_STRANGLE_FALLBACK = False   # legacy paired constructor disabled
ENABLE_ASYMMETRIC_CARRY = True
STRANGLE_MIN_IV_RV = 1.00          # retained as the per-leg absolute-IV/RV gate
STRANGLE_TARGET_DELTA = 0.16
STRANGLE_MIN_ABS_DELTA = 0.08
STRANGLE_MAX_ABS_DELTA = 0.25
STRANGLE_MAX_CANDIDATES = 12
STRANGLE_USE_CONTINUOUS_RV = True

# Uncapped risk proxy remains the same full-package proxy used by BOLD_V4.
UNCAPPED_MARGIN_NOTIONAL_PCT = 0.12
UNCAPPED_MAX_SINGLE_TRADE_RISK_PCT = 7.5
UNCAPPED_MAX_TICKER_RISK_PCT = 7.5
UNCAPPED_POSITION_PCT_PER_ACTION = 7.5
UNCAPPED_MAX_CONTRACTS_PER_TRADE = 1

# Side-specific HJB/execution budgets.  These are FROZEN for WF_A/B/C/D.
# PUT keeps the full package-style risk basis and receives the larger uncapped budget.
# CALL receives exactly half that capital budget because the ablation showed the
# upside leg was the dominant loss source.  Calls are still allowed if their own MC/HJB
# economics independently justify the trade.
ASYM_PUT_POSITION_PCT_PER_ACTION = 7.5
ASYM_PUT_MAX_SINGLE_TRADE_RISK_PCT = 7.5
ASYM_PUT_MAX_TICKER_RISK_PCT = 7.5
ASYM_PUT_MAX_CONTRACTS_PER_TRADE = 2

ASYM_CALL_POSITION_PCT_PER_ACTION = 3.75
ASYM_CALL_MAX_SINGLE_TRADE_RISK_PCT = 3.75
ASYM_CALL_MAX_TICKER_RISK_PCT = 3.75
ASYM_CALL_MAX_CONTRACTS_PER_TRADE = 1

# If both independently pass, the resulting strangle is call-limited rather than
# silently restoring the larger put budget.
ASYM_STRANGLE_POSITION_PCT_PER_ACTION = ASYM_CALL_POSITION_PCT_PER_ACTION
ASYM_STRANGLE_MAX_SINGLE_TRADE_RISK_PCT = ASYM_CALL_MAX_SINGLE_TRADE_RISK_PCT
ASYM_STRANGLE_MAX_TICKER_RISK_PCT = ASYM_CALL_MAX_TICKER_RISK_PCT
ASYM_STRANGLE_MAX_CONTRACTS_PER_TRADE = 1

# Carry legs use package-level management only.  This is the live/MC analogue of the
# C3 ablation: no 15% individual-leg stop/trail for naked carry positions.
ASYM_PACKAGE_ONLY_MANAGEMENT = True

# -------- V7 scope freeze --------
# Cross-window ablations showed standalone short calls were the dominant loss source.
# Until a direction-aware underlying-return model is added, a CALL may only be used
# when the same ticker/expiry PUT is independently selected, creating a strangle.
ASYM_ALLOW_STANDALONE_CALL = False
# The legacy SABR/local-fly branch remains computed for diagnostics, but execution is
# disabled because the multi-window ablation showed negative realized contribution.
ENABLE_LEGACY_FLY_EXECUTION = False

# -------- BOLD_V4: stepped live management --------
MANAGEMENT_FREQUENCY = 'DAILY_CLOSE'
MANAGE_INTERVAL_MINUTES = 5  # legacy only; V6 does not run intraday management
LEG_STOP_LOSS_PCT = 0.15
STRATEGY_STOP_TRIGGER_PCT = 0.08
STRATEGY_HARD_STOP_PCT = 0.10
LEG_TRAIL_START_PCT = 0.25
STRATEGY_TRAIL_START_PCT = 0.15
TRAIL_STEP_PCT = 0.10

FORCE_FLAT_END = True
MAX_NEW_TRADES_PER_DAY = 5
MAX_CONTRACTS_PER_TRADE = 5

# No Object Store dependency in the validation build.
SAVE_RESEARCH_SNAPSHOTS = False
OBJECT_STORE_PREFIX = 'vol_thesis_v3'
