from AlgorithmImports import *
# endregion

# Your New Python File
import math
import numpy as np

RISK_FREE_RATE = 0.04

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def safe_float(x, default=np.nan):
    try:
        if x is None or np.isnan(float(x)):
            return default
        return float(x)
    except Exception:
        return default

def black_scholes_greeks(S: float, K: float, T: float, sigma: float, option_type: str, r: float = RISK_FREE_RATE) -> dict:
    S = safe_float(S); K = safe_float(K); T = safe_float(T); sigma = safe_float(sigma)
    if np.isnan(S) or np.isnan(K) or np.isnan(T) or np.isnan(sigma) or S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {"delta": np.nan,"gamma": np.nan,"vega": np.nan,"theta": np.nan,"vanna": np.nan,"volga": np.nan,"charm": np.nan}
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = norm_pdf(d1)
    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT / 100.0
    volga = vega * d1 * d2 / sigma
    vanna = -pdf * d2 / sigma / 100.0
    if option_type == "call":
        delta = norm_cdf(d1)
        theta = (-S * pdf * sigma / (2.0 * sqrtT) - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365.0
        charm = (-pdf * ((2.0 * r * T - d2 * sigma * sqrtT) / (2.0 * T * sigma * sqrtT))) / 365.0
    else:
        delta = norm_cdf(d1) - 1.0
        theta = (-S * pdf * sigma / (2.0 * sqrtT) + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365.0
        charm = (-pdf * ((2.0 * r * T - d2 * sigma * sqrtT) / (2.0 * T * sigma * sqrtT))) / 365.0
    return {"delta": float(delta),"gamma": float(gamma),"vega": float(vega),"theta": float(theta),"vanna": float(vanna),"volga": float(volga),"charm": float(charm)}
