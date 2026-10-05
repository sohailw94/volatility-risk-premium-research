from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
ACTION_SIZES = np.array([0.0, 0.25, 0.50, 0.75, 1.00])

BASE_POSITION_PCT = 5.0

LAMBDA_MAX_LOSS = 0.70
LAMBDA_DELTA = 0.35
LAMBDA_VEGA = 0.20
LAMBDA_MARGIN = 0.15
LAMBDA_BAD_STRUCTURE = 0.50

MAX_ABS_DELTA = 0.15


def safe_float(row, col, default=0.0):
    val = row.get(col, default)
    if pd.isna(val):
        return default
    try:
        return float(val)
    except Exception:
        return default


def vol_edge(row):
    ratio = safe_float(row, "variance_premium_ratio", 1.0)
    xs_score = safe_float(row, "cross_sectional_vrp_score", 0.0)
    side = row.get("vol_side", "NONE")

    if side == "SHORT_VOL":
        raw_edge = ratio - 1.0
    elif side == "LONG_VOL":
        raw_edge = 1.0 - ratio
    elif side == "PIN_RANGE":
        raw_edge = abs(ratio - 1.0) * 0.50
    else:
        raw_edge = 0.0

    return max(0.0, raw_edge) + 0.25 * max(0.0, abs(xs_score))


def structure_quality(row):
    structure = row.get("structure", "")
    max_loss = safe_float(row, "max_loss", np.nan)
    max_profit = safe_float(row, "max_profit", np.nan)
    net_delta = abs(safe_float(row, "net_delta", 0.0))

    if structure == "" or structure == "NO_TRADE":
        return 0.0

    if pd.isna(max_loss) or max_loss <= 0:
        return 0.0

    reward_risk = max_profit / max_loss if max_loss > 0 else 0.0
    reward_risk = np.clip(reward_risk, 0.0, 5.0) / 5.0

    delta_quality = max(0.0, 1.0 - net_delta / MAX_ABS_DELTA)

    return 0.60 * reward_risk + 0.40 * delta_quality


def risk_penalty(row):
    max_loss = safe_float(row, "max_loss", 0.0)
    net_delta = abs(safe_float(row, "net_delta", 0.0))
    net_vega = abs(safe_float(row, "net_vega", 0.0))

    max_loss_scaled = max_loss / 10.0
    delta_scaled = net_delta / MAX_ABS_DELTA
    vega_scaled = net_vega / 10.0

    return (
        LAMBDA_MAX_LOSS * max_loss_scaled
        + LAMBDA_DELTA * delta_scaled
        + LAMBDA_VEGA * vega_scaled
    )


def utility(row, action_size):
    mc_expected_pnl = row.get("mc_expected_pnl", 0.0) / 100.0
    mc_pop = row.get("mc_pop", 0.0)
    mc_sharpe = row.get("mc_expected_sharpe", 0.0)
    mc_barrier = row.get("mc_barrier_prob", 0.0)
    mc_cvar = row.get("mc_cvar_95", 0.0) / 100.0
    mc_max_loss_prob = row.get("mc_max_loss_prob", 0.0)

    structure_quality_score = row.get("structure_quality_score", 0.0)
    vrp_score = row.get("cross_sectional_vrp_score", 0.0)

    delta_penalty = abs(row.get("net_delta", 0.0))
    vega_penalty = abs(row.get("net_vega", 0.0))
    theta_penalty = max(0.0, -row.get("net_theta", 0.0))

    gross_reward = (
        0.25 * mc_expected_pnl
        + 0.20 * mc_sharpe
        + 0.15 * mc_pop
        + 0.15 * mc_barrier
        + 0.15 * structure_quality_score
        + 0.10 * vrp_score
    )

    risk_penalty = (
        0.40 * abs(mc_cvar)
        + 0.35 * mc_max_loss_prob
        + 0.20 * delta_penalty
        + 0.20 * vega_penalty
        + 0.10 * theta_penalty
        + 0.15 * action_size
    )

    return action_size * gross_reward - (action_size ** 2) * risk_penalty


def optimize_row(row):
    values = np.array([utility(row, a) for a in ACTION_SIZES])
    idx = int(np.argmax(values))

    return float(ACTION_SIZES[idx]), float(values[idx])


def classify_decision(action_size):
    if action_size == 0:
        return "NO_TRADE"
    if action_size <= 0.25:
        return "TRADE_TINY"
    if action_size <= 0.50:
        return "TRADE_SMALL"
    return "TRADE_NORMAL"


def run_wealth_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:return pd.DataFrame() if df is None else df.copy()
    rows=[]
    for _,row in df.iterrows():
        out=row.to_dict()
        # Preserve original ordering exactly: optimize_row sees no newly-computed
        # structure_quality_score/risk_penalty_score unless they already existed upstream.
        action_size,value=optimize_row(row)
        out["vol_edge_score"]=vol_edge(row);out["structure_quality_score"]=structure_quality(row);out["risk_penalty_score"]=risk_penalty(row)
        out["wealth_action_size"]=action_size;out["wealth_optimizer_value"]=value;out["optimized_position_pct"]=round(BASE_POSITION_PCT*action_size,2);out["wealth_decision"]=classify_decision(action_size)
        rows.append(out)
    return pd.DataFrame(rows).sort_values("wealth_optimizer_value",ascending=False).reset_index(drop=True)


