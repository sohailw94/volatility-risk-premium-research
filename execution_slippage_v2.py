# region imports
from AlgorithmImports import *
# endregion

import json
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_MULTIPLIER = 100.0
MIN_DOLLAR_COST_PER_LEG = 0.50
BASE_COMPLEX_ORDER_CAPTURE = 0.35
SIZE_IMPACT_PER_EXTRA_CONTRACT = 0.05


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def parse_legs(value: Any) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError("legs_json must be a list or JSON string.")


def signed_quantity(leg: dict) -> int:
    signed = int(leg.get("signed_qty", 0) or 0)
    if signed:
        return signed
    quantity = int(leg.get("qty", 1) or 1)
    return quantity if str(leg.get("side", "")).upper() == "BUY" else -quantity


def regime_multiplier(row: pd.Series | dict) -> float:
    value = _safe_float(row.get("hmm_slippage_multiplier", 1.0), 1.0)
    return float(np.clip(value, 0.5, 3.0))


def liquidity_multiplier(
    open_interest: float | None,
    volume: float | None,
    spread_pct: float | None,
) -> float:
    oi = max(_safe_float(open_interest, 0.0), 0.0)
    vol = max(_safe_float(volume, 0.0), 0.0)
    spread = max(_safe_float(spread_pct, 0.0), 0.0)

    multiplier = 1.0
    if oi < 100:
        multiplier += 0.35
    elif oi < 500:
        multiplier += 0.15

    if vol < 10:
        multiplier += 0.30
    elif vol < 50:
        multiplier += 0.10

    multiplier += min(spread, 1.0) * 0.50
    return float(multiplier)


def estimate_leg_half_spread_dollars(leg: dict) -> float:
    bid = _safe_float(leg.get("bid"))
    ask = _safe_float(leg.get("ask"))

    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
        half_spread = 0.5 * (ask - bid)
    else:
        mid = _safe_float(leg.get("mid"), 0.0)
        spread_pct = max(_safe_float(leg.get("spread_pct"), 0.10), 0.0)
        half_spread = 0.5 * abs(mid) * spread_pct

    quantity = abs(signed_quantity(leg))
    return max(
        half_spread * quantity * CONTRACT_MULTIPLIER,
        MIN_DOLLAR_COST_PER_LEG * quantity,
    )


def estimate_strategy_slippage(
    row: pd.Series | dict,
    contracts: int = 1,
    phase: str = "ENTRY",
) -> float:
    """
    Estimate one-way package execution drag in dollars.

    phase:
      ENTRY, EXIT, or STOP_EXIT
    """
    legs = parse_legs(row["legs_json"])
    raw_half_spread = sum(estimate_leg_half_spread_dollars(leg) for leg in legs)

    phase = phase.upper()
    phase_multiplier = {
        "ENTRY": 1.00,
        "EXIT": 1.15,
        "STOP_EXIT": 1.50,
    }.get(phase, 1.00)

    size_multiplier = 1.0 + SIZE_IMPACT_PER_EXTRA_CONTRACT * max(contracts - 1, 0)

    # Complex-order execution generally captures only part of the sum of leg
    # half-spreads. Stress regimes reduce that advantage via regime_multiplier.
    package_capture = BASE_COMPLEX_ORDER_CAPTURE
    cost = (
        raw_half_spread
        * package_capture
        * regime_multiplier(row)
        * phase_multiplier
        * size_multiplier
        * max(int(contracts), 1)
    )
    return float(max(cost, 0.0))


def round_trip_slippage(
    row: pd.Series | dict,
    contracts: int = 1,
    stop_exit_probability: float = 0.0,
) -> float:
    entry = estimate_strategy_slippage(row, contracts, "ENTRY")
    normal_exit = estimate_strategy_slippage(row, contracts, "EXIT")
    stop_exit = estimate_strategy_slippage(row, contracts, "STOP_EXIT")
    probability = float(np.clip(stop_exit_probability, 0.0, 1.0))
    expected_exit = (1.0 - probability) * normal_exit + probability * stop_exit
    return float(entry + expected_exit)


