from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
import qc_config_v2 as cfg


def _finite(x):
    a = np.asarray(x, float)
    return a[np.isfinite(a)]


def _ann_var(x):
    a = _finite(x)
    return np.nan if len(a) < 2 else float(np.var(a, ddof=0) * cfg.TRADING_DAYS)


def _ewma_var(x, lam=None):
    lam = cfg.EWMA_LAMBDA if lam is None else float(lam)
    a = _finite(x)
    if len(a) == 0:
        return np.nan
    v = float(a[0] ** 2)
    for z in a[1:]:
        v = lam * v + (1.0 - lam) * float(z) ** 2
    return float(v * cfg.TRADING_DAYS)


def _target_expiry(surface_ticker):
    if surface_ticker is None or surface_ticker.empty:
        return None
    x = surface_ticker[['expiry','dte']].drop_duplicates().copy()
    if x.empty:
        return None
    i = (x['dte'].astype(float) - cfg.TARGET_DTE).abs().idxmin()
    return pd.Timestamp(x.loc[i, 'expiry'])


def _event_var_proxy(surface_ticker, target_expiry):
    """ATM total-variance bump around target expiry.

    This uses only the contemporaneous option term structure.  It is an event
    variance proxy, not an earnings-calendar lookup.
    """
    if surface_ticker is None or surface_ticker.empty or target_expiry is None:
        return 0.0
    terms = []
    for expiry, g in surface_ticker.groupby('expiry'):
        z = g.copy()
        spot = max(float(z['underlying_price'].median()), 1e-9)
        z['md'] = np.abs(np.log(z['strike'].astype(float) / spot))
        iv = pd.to_numeric(z.sort_values('md').head(4)['implied_volatility'], errors='coerce').dropna()
        if len(iv) == 0:
            continue
        dte = int(z['dte'].median())
        T = max(dte / 365.0, 1.0 / 365.0)
        atm = float(iv.median())
        terms.append((pd.Timestamp(expiry), T, atm * atm * T))
    terms.sort(key=lambda x: x[1])
    for i, (expiry, T, w) in enumerate(terms):
        if expiry != pd.Timestamp(target_expiry):
            continue
        if 0 < i < len(terms) - 1:
            _, t0, w0 = terms[i - 1]
            _, t1, w1 = terms[i + 1]
            if t1 > t0:
                baseline = w0 + (w1 - w0) * (T - t0) / (t1 - t0)
                return float(max(w - baseline, 0.0) / max(T, 1e-9))
        return 0.0
    return 0.0


def build_forward_vol_features(prices, surface=None, as_of=None):
    rows = []
    surface = pd.DataFrame() if surface is None else surface.copy()
    kappa = np.log(2.0) / float(cfg.OU_VARIANCE_HALFLIFE)

    for ticker, g in prices.groupby('ticker'):
        g = g.sort_values('date')
        r = np.log(g['close'].astype(float)).diff().dropna().to_numpy(float)
        if len(r) < 60:
            continue

        raw20 = r[-20:]
        raw60 = r[-60:]
        raw120 = r[-min(len(r), int(cfg.JUMP_LOOKBACK)):]
        rv20 = _ann_var(raw20)
        rv60 = _ann_var(raw60)
        ew = _ewma_var(raw120)
        if not all(np.isfinite(v) for v in [rv20, rv60, ew]):
            continue

        st = surface[surface['ticker'] == ticker].copy() if not surface.empty else pd.DataFrame()
        exp = _target_expiry(st)
        if exp is not None and not st.empty:
            tgt = st[st['expiry'] == exp]
            dte = int(tgt['dte'].median()) if not tgt.empty else cfg.TARGET_DTE
        else:
            dte = cfg.TARGET_DTE
        horizon_td = max(int(round(dte * cfg.TRADING_DAYS / 365.0)), 1)

        # Original OU mean reversion on variance.
        ou_var = rv60 + (ew - rv60) * np.exp(-kappa * horizon_td)
        original_ou_var = cfg.FORWARD_OU_WEIGHT * ou_var + cfg.FORWARD_EWMA_WEIGHT * ew

        # Separate continuous variance from historical jumps.
        daily_sigma = float(np.std(raw60, ddof=0))
        cap = float(cfg.JUMP_Z) * max(daily_sigma, 1e-9)
        clipped = np.clip(raw120, -cap, cap)
        cont60 = _ann_var(clipped[-min(60, len(clipped)):])
        cont_ew = _ewma_var(clipped)
        cont_ou = cont60 + (cont_ew - cont60) * np.exp(-kappa * horizon_td)
        continuous_var = cfg.FORWARD_OU_WEIGHT * cont_ou + cfg.FORWARD_EWMA_WEIGHT * cont_ew

        jump_excess_sq = np.maximum(raw120**2 - clipped**2, 0.0)
        jump_var = float(np.mean(jump_excess_sq) * cfg.TRADING_DAYS)
        event_var = _event_var_proxy(st, exp)

        forecast_var = float(max(continuous_var + jump_var + event_var, 1e-12))
        rows.append({
            'date': pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else g['date'].max(),
            'ticker': ticker,
            'target_expiry': exp,
            'target_dte': int(dte),
            'forecast_horizon_td': int(horizon_td),
            'rv_20d_var': float(rv20),
            'rv_60d_var': float(rv60),
            'ewma_var': float(ew),
            'original_ou_variance_20d': float(max(original_ou_var, 1e-12)),
            'continuous_variance_20d': float(max(continuous_var, 1e-12)),
            'jump_variance_20d': float(max(jump_var, 0.0)),
            'event_variance_20d': float(max(event_var, 0.0)),
            'forecast_variance_20d': forecast_var,
            'forecast_vol_20d': float(np.sqrt(forecast_var)),
            'forecast_model': 'THESIS_V3_CONTINUOUS_OU_PLUS_JUMP_PLUS_TERM_EVENT',
        })
    return pd.DataFrame(rows)
