import logging
import warnings

import numpy as np
import pandas as pd

try:
    import qc_config_v2 as cfg
except Exception:
    cfg = None

FEATURES = [
    'return_1d',
    'rv_20d',
    'vol_of_vol_20d',
    'downside_20d',
    'trend_20d',
    'absolute_return_5d',
]


def _cfg(name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def latest_regimes_from_prices(prices, min_observations=None):
    """Point-in-time, in-memory HMM regime estimator for QuantConnect."""
    min_observations = int(
        min_observations
        if min_observations is not None
        else _cfg('HMM_MIN_OBSERVATIONS', _cfg('MIN_HISTORY_ROWS', 80))
    )
    n_states = int(_cfg('HMM_N_STATES', 3))
    random_state = int(_cfg('HMM_RANDOM_STATE', 42))

    if prices is None or len(prices) == 0:
        return pd.DataFrame()

    rows = []
    for ticker, g in prices.groupby('ticker'):
        g = g.sort_values('date').copy()
        close = pd.to_numeric(g['close'], errors='coerce').clip(lower=1e-9)
        lp = np.log(close)
        r = lp.diff()
        rv5 = r.rolling(5).std(ddof=0) * np.sqrt(252.0)

        g['return_1d'] = r
        g['rv_20d'] = r.rolling(20).std(ddof=0) * np.sqrt(252.0)
        g['vol_of_vol_20d'] = rv5.rolling(20).std(ddof=0)
        g['downside_20d'] = r.where(r < 0, 0.0).rolling(20).std(ddof=0) * np.sqrt(252.0)
        g['trend_20d'] = lp.diff(20)
        g['absolute_return_5d'] = r.abs().rolling(5).mean()
        g = g.dropna(subset=FEATURES)
        if len(g) < min_observations:
            continue

        x = g[FEATURES].astype(float).copy()
        for col in FEATURES:
            sd = float(x[col].std(ddof=0))
            x[col] = 0.0 if (not np.isfinite(sd) or sd <= 1e-12) else (x[col] - float(x[col].mean())) / sd

        X = x.to_numpy(float)
        model_name = 'QUANTILE_REGIME_FALLBACK'
        try:
            from hmmlearn.hmm import GaussianHMM
            m = GaussianHMM(
                n_components=n_states,
                covariance_type='diag',
                n_iter=300,
                tol=1e-4,
                random_state=random_state,
            )

            hmm_logger = logging.getLogger('hmmlearn.base')
            old_level = hmm_logger.level
            try:
                hmm_logger.setLevel(logging.ERROR)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m.fit(X)
            finally:
                hmm_logger.setLevel(old_level)

            states = m.predict(X)
            probs = m.predict_proba(X)
            model_name = 'GAUSSIAN_HMM'
        except Exception:
            try:
                from sklearn.mixture import GaussianMixture
                m = GaussianMixture(
                    n_components=n_states,
                    covariance_type='diag',
                    n_init=8,
                    random_state=random_state,
                )
                states = m.fit_predict(X)
                probs = m.predict_proba(X)
                model_name = 'GMM_FALLBACK'
            except Exception:
                risk = X[:, 1] + X[:, 2] + X[:, 3] + X[:, 5] - X[:, 0]
                q1, q2 = np.quantile(risk, [1.0 / 3.0, 2.0 / 3.0])
                states = np.where(risk <= q1, 0, np.where(risk >= q2, 2, 1))
                probs = np.zeros((len(risk), 3), dtype=float)
                probs[np.arange(len(risk)), states] = 1.0

        risk_score = x['rv_20d'] + x['vol_of_vol_20d'] + x['downside_20d'] + x['absolute_return_5d'] - x['return_1d']
        means = {s: float(risk_score[states == s].mean()) if np.any(states == s) else np.nan for s in range(3)}
        observed = sorted([s for s in means if np.isfinite(means[s])], key=lambda s: means[s])
        missing = [s for s in range(3) if s not in observed]
        order = (observed + missing)[:3]
        labels = {s: lab for s, lab in zip(order, ['CALM', 'NORMAL', 'STRESS'])}
        for s in range(3):
            labels.setdefault(s, 'NORMAL')

        p = {'CALM': 0.0, 'NORMAL': 0.0, 'STRESS': 0.0}
        last = len(g) - 1
        for s in range(3):
            p[labels[s]] += float(probs[last, s])
        tot = sum(p.values()) or 1.0
        p = {k: v / tot for k, v in p.items()}

        rows.append({
            'date': g['date'].iloc[-1],
            'ticker': ticker,
            'hmm_regime': max(p, key=p.get),
            'hmm_prob_calm': p['CALM'],
            'hmm_prob_normal': p['NORMAL'],
            'hmm_prob_stress': p['STRESS'],
            'hmm_jump_multiplier': p['CALM'] * 0.50 + p['NORMAL'] * 1.00 + p['STRESS'] * 2.50,
            'hmm_factor_loading': p['CALM'] * 0.25 + p['NORMAL'] * 0.40 + p['STRESS'] * 0.70,
            'hmm_slippage_multiplier': p['CALM'] * 0.80 + p['NORMAL'] * 1.00 + p['STRESS'] * 1.75,
            'hmm_model': model_name,
        })

    return pd.DataFrame(rows)
