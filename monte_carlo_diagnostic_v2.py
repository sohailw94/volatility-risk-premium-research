# region imports
from AlgorithmImports import *
# endregion

import hashlib
import json
import numpy as np
import pandas as pd
import qc_config_v2 as cfg


def _trade_id(row):
    payload = '|'.join(map(str, [
        row.get('snapshot_date'),
        row.get('ticker'),
        row.get('structure'),
        row.get('expiry'),
        row.get('legs_json'),
        row.get('net_credit'),
        row.get('net_debit'),
    ]))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _finite_or(value, default):
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _payoff(option_type, strike, terminal_spot):
    terminal_spot = np.asarray(terminal_spot, dtype=float)
    if str(option_type).lower().startswith('c'):
        return np.maximum(terminal_spot - float(strike), 0.0)
    return np.maximum(float(strike) - terminal_spot, 0.0)


def _terminal_pnl(row, terminal_spot):
    legs = (
        json.loads(row['legs_json'])
        if isinstance(row['legs_json'], str)
        else row['legs_json']
    )

    terminal_value = np.zeros_like(np.asarray(terminal_spot, dtype=float))

    for leg in legs:
        terminal_value += int(leg['signed_qty']) * _payoff(
            leg['option_type'],
            float(leg['strike']),
            terminal_spot,
        )

    if str(row['premium_type']).upper() == 'CREDIT':
        return (
            terminal_value + abs(float(row['net_credit']))
        ) * cfg.CONTRACT_MULTIPLIER

    return (
        terminal_value - abs(float(row['net_debit']))
    ) * cfg.CONTRACT_MULTIPLIER


def _business_days(snapshot, expiry, max_days):
    snapshot = pd.Timestamp(snapshot).normalize()
    expiry = pd.Timestamp(expiry).normalize()

    # Business-day approximation. It removes weekends but not exchange holidays.
    days = len(
        pd.bdate_range(
            snapshot,
            expiry,
            inclusive='right'
        )
    )

    return min(max(days, 1), int(max_days))


def run_mc_clean_diagnostic(strategies, n_paths=None):
    """
    Diagnostic MC only.

    Differences versus the production/current MC:
    1. Gaussian shocks instead of Student-t shocks.
    2. No explicit jump process.
    3. Business-day horizon instead of calendar-day horizon.
    4. Same forecast_vol_20d input and same terminal payoff calculation.

    This is intended to isolate how much EV inflation is coming from
    heavy tails + jumps + calendar-day horizon. It should NOT automatically
    replace the production MC until the comparison is reviewed.
    """
    n_paths = int(n_paths or cfg.N_PATHS)

    if strategies is None or strategies.empty:
        return pd.DataFrame()

    data = strategies.copy()
    rows = []

    snapshot = pd.Timestamp(
        data['snapshot_date'].iloc[0]
    ).normalize()

    common_rng = np.random.default_rng(
        int(snapshot.strftime('%Y%m%d')) + 991
    )

    max_days = int(cfg.MC_MAX_DAYS)

    # Common Gaussian market factor.
    common = common_rng.standard_normal(
        size=(n_paths, max_days)
    )

    for _, row in data.iterrows():
        days = _business_days(
            snapshot,
            row['expiry'],
            max_days,
        )

        sigma = _finite_or(
            row.get('forecast_vol_20d'),
            _finite_or(
                row.get('atm_iv_20d'),
                0.50
            ),
        )
        sigma = float(
            np.clip(sigma, 0.03, 3.0)
        )

        loading = float(
            np.clip(
                _finite_or(
                    row.get('hmm_factor_loading'),
                    0.40
                ),
                0.0,
                0.95,
            )
        )

        trade_id = _trade_id(row)

        seed = int(
            hashlib.sha256(
                ('CLEAN|' + trade_id).encode()
            ).hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(seed)

        idiosyncratic = rng.standard_normal(
            size=(n_paths, days)
        )

        z = (
            loading * common[:, :days]
            + np.sqrt(
                max(
                    1.0 - loading * loading,
                    0.0
                )
            ) * idiosyncratic
        )

        dt = 1.0 / float(cfg.TRADING_DAYS)

        log_returns = (
            -0.5 * sigma * sigma * dt
            + sigma * np.sqrt(dt) * z
        )

        terminal_spot = float(
            row['underlying_price']
        ) * np.exp(
            log_returns.sum(axis=1)
        )

        pnl = _terminal_pnl(
            row,
            terminal_spot,
        )

        q05 = float(
            np.quantile(pnl, 0.05)
        )
        tail = pnl[pnl <= q05]

        output = row.to_dict()
        output.update({
            'mc_trade_id': trade_id,
            'diag_business_days': days,
            'diag_sigma': sigma,
            'diag_expected_pnl': float(
                np.mean(pnl)
            ),
            'diag_median_pnl': float(
                np.median(pnl)
            ),
            'diag_pop': float(
                np.mean(pnl > 0)
            ),
            'diag_cvar_95': float(
                -np.mean(tail)
            ) if len(tail) else 0.0,
            'diag_expected_sharpe': float(
                np.mean(pnl)
                / (np.std(pnl) + 1e-12)
            ),
            'diag_model':
                'GAUSSIAN_NO_JUMPS_BUSINESS_DAYS',
        })

        rows.append(output)

    return pd.DataFrame(rows)


def compare_mc_evs(current_mc, clean_mc):
    """
    Returns one row per candidate with current-vs-clean EV diagnostics.
    """
    if (
        current_mc is None
        or current_mc.empty
        or clean_mc is None
        or clean_mc.empty
    ):
        return pd.DataFrame()

    left_cols = [
        'mc_trade_id',
        'ticker',
        'structure',
        'mc_expected_pnl',
        'mc_pop',
        'mc_cvar_95',
        'mc_expected_sharpe',
    ]

    right_cols = [
        'mc_trade_id',
        'diag_business_days',
        'diag_sigma',
        'diag_expected_pnl',
        'diag_pop',
        'diag_cvar_95',
        'diag_expected_sharpe',
    ]

    left = current_mc[
        [c for c in left_cols if c in current_mc.columns]
    ].copy()

    right = clean_mc[
        [c for c in right_cols if c in clean_mc.columns]
    ].copy()

    out = left.merge(
        right,
        on='mc_trade_id',
        how='inner',
    )

    out['ev_change'] = (
        out['diag_expected_pnl']
        - out['mc_expected_pnl']
    )

    out['ev_ratio_clean_to_current'] = np.where(
        np.abs(out['mc_expected_pnl']) > 1e-12,
        out['diag_expected_pnl']
        / out['mc_expected_pnl'],
        np.nan,
    )

    return out
