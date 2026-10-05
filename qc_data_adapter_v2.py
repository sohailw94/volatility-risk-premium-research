
from datetime import timedelta
import numpy as np
import pandas as pd
from AlgorithmImports import Resolution
import qc_config_v2 as cfg


def _symbol_value(x):
    try:
        return str(x.value)
    except Exception:
        return str(x)


def history_to_price_frame(algorithm, equity_symbols, lookback_bars):
    """One bulk point-in-time daily history request."""
    symbols = list(equity_symbols.values())
    if not symbols:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])
    start = algorithm.time - timedelta(days=int(max(lookback_bars, 120) * 1.8))
    try:
        h = algorithm.history(symbols, start, algorithm.time, Resolution.DAILY)
    except Exception:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])
    if h is None or len(h) == 0:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])

    f = h.reset_index()
    dc = next((c for c in f.columns if 'time' in str(c).lower() or str(c).lower() == 'date'), None)
    sc = next((c for c in f.columns if 'symbol' in str(c).lower()), None)
    cc = next((c for c in f.columns if str(c).lower() == 'close'), None)
    if dc is None or sc is None or cc is None:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])

    reverse = {_symbol_value(sym): ticker for ticker, sym in equity_symbols.items()}
    f['ticker'] = f[sc].map(lambda x: reverse.get(_symbol_value(x)))
    f['date'] = pd.to_datetime(f[dc], errors='coerce').dt.tz_localize(None).dt.normalize()
    f['close'] = pd.to_numeric(f[cc], errors='coerce')
    f = f.dropna(subset=['ticker', 'date', 'close'])[['date', 'ticker', 'close']]

    # Strict point-in-time hygiene.  The research cycle runs at 10:00 ET; a
    # current-session daily bar must never be allowed into the RV/HMM history.
    # This is a plumbing/data-validity correction only, not a signal threshold.
    today = pd.Timestamp(algorithm.time).tz_localize(None).normalize()
    f = f[f['date'] < today].copy()

    return (
        f.sort_values(['ticker', 'date'])
        .drop_duplicates(['ticker', 'date'], keep='last')
        .reset_index(drop=True)
    )


def current_option_surface(algorithm, data, option_symbols, equity_symbols, min_dte, max_dte, max_relative_spread=None):
    """Build a tradable surface from the CURRENT Slice.

    V3.2 data-validity fix:
    - IV observations are admitted only with a live TWO-SIDED quote.
    - No last-price fallback is used for the surface.
    - An optional mechanical relative-spread cap can remove unusable quotes
      before they contaminate SABR/VRP.  This is a liquidity rule, not alpha.
    - Actual executable package economics are still enforced downstream.

    The V3.1 walk-forward implementation admitted contracts with bid=0 and used
    last price as the mark.  That can manufacture apparent surface residuals
    which cannot actually be sold/bought at the observed quote.
    """
    rows = []
    today = pd.Timestamp(algorithm.time).tz_localize(None).normalize()

    for ticker, canonical in option_symbols.items():
        try:
            chain = data.option_chains.get(canonical)
        except Exception:
            chain = None
        if chain is None:
            continue

        try:
            spot = float(algorithm.securities[equity_symbols[ticker]].price)
        except Exception:
            continue
        if not np.isfinite(spot) or spot <= 0:
            continue

        for c in chain:
            try:
                expiry = pd.Timestamp(c.expiry).tz_localize(None).normalize()
                dte = int((expiry - today).days)
                if not min_dte <= dte <= max_dte:
                    continue

                iv = float(c.implied_volatility)
                bid = float(c.bid_price or 0.0)
                ask = float(c.ask_price or 0.0)

                # Mechanical data-validity gate, not an alpha threshold.
                if not np.isfinite(iv) or iv <= 0:
                    continue
                if not (np.isfinite(bid) and np.isfinite(ask)):
                    continue
                if bid <= 0.0 or ask <= 0.0 or ask < bid:
                    continue

                mid = 0.5 * (bid + ask)
                if mid <= 0.0:
                    continue
                spread = ask - bid
                rel_spread = spread / mid
                if max_relative_spread is not None and rel_spread > float(max_relative_spread):
                    if spread > float(getattr(cfg,'LIQUIDITY_ABSOLUTE_SPREAD_OVERRIDE',0.03)):
                        continue

                right = 'call' if str(c.right).lower().endswith('call') else 'put'
                rows.append({
                    'snapshot_date': today,
                    'snapshot_timestamp': pd.Timestamp(algorithm.time).tz_localize(None),
                    'ticker': ticker,
                    'symbol': c.symbol,
                    'expiry': expiry,
                    'dte': dte,
                    'strike': float(c.strike),
                    'option_type': right,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'bid_ask_spread': spread,
                    'relative_bid_ask_spread': rel_spread,
                    'implied_volatility': iv,
                    'underlying_price': spot,
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(['ticker', 'expiry', 'option_type', 'strike'])
        .reset_index(drop=True)
    )


