# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
import qc_config_v2 as cfg


def estimate_ou_params(series: pd.Series) -> dict:
    x = series.dropna().astype(float).clip(cfg.OU_IV_MIN, cfg.OU_IV_MAX)
    if len(x) < cfg.OU_IV_MIN_OBSERVATIONS:
        return {"ou_kappa":np.nan,"ou_theta":np.nan,"ou_eta":np.nan,"ou_half_life":np.nan,"ou_forecast_iv":np.nan,"ou_expected_iv_change":np.nan,"ou_zscore":np.nan}
    y = x.shift(-1).dropna(); x_lag = x.iloc[:-1]
    if len(x_lag) < cfg.OU_IV_MIN_OBSERVATIONS - 1:
        return {}
    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    a,b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    b = np.clip(b, 0.01, 0.999)
    kappa = -np.log(b); theta = a/(1.0-b)
    residuals = y.values - (a + b*x_lag.values); eta = np.std(residuals)
    latest = x.iloc[-1]; forecast = theta + b*(latest-theta)
    half_life = np.log(2.0)/kappa if kappa > 0 else np.nan
    zscore = (latest-theta)/eta if eta > 0 else 0.0
    return {"ou_kappa":float(kappa),"ou_theta":float(theta),"ou_eta":float(eta),"ou_half_life":float(half_life),"ou_forecast_iv":float(forecast),"ou_expected_iv_change":float(forecast-latest),"ou_zscore":float(zscore)}


def _current_atm(surface):
    rows=[]
    for ticker,g in surface.groupby("ticker"):
        spot=float(g["underlying_price"].median()); q=g.copy(); q["moneyness_dist"]=(q["strike"].astype(float)-spot).abs()
        atm=q.sort_values("moneyness_dist").head(10); iv=pd.to_numeric(atm["implied_volatility"],errors="coerce").dropna()
        if not iv.empty: rows.append({"ticker":ticker,"atm_iv":float(iv.mean())})
    return pd.DataFrame(rows)


def update_ou_features(algorithm, surface: pd.DataFrame) -> pd.DataFrame:
    if surface is None or surface.empty:return pd.DataFrame()
    today=pd.Timestamp(algorithm.time).normalize(); atm=_current_atm(surface)
    if not hasattr(algorithm,"_vh_iv_history"): algorithm._vh_iv_history={}
    rows=[]
    for _,r in atm.iterrows():
        ticker=str(r.ticker); hist=algorithm._vh_iv_history.setdefault(ticker,[])
        if hist and pd.Timestamp(hist[-1][0]).normalize()==today: hist[-1]=(today,float(r.atm_iv))
        else: hist.append((today,float(r.atm_iv)))
        if len(hist)>cfg.OU_IV_MAX_HISTORY: del hist[:-cfg.OU_IV_MAX_HISTORY]
        series=pd.Series([x[1] for x in hist],dtype=float); params=estimate_ou_params(series)
        rows.append({"ticker":ticker,"snapshot_date":today,"current_atm_iv":float(r.atm_iv),**params})
    return pd.DataFrame(rows)

