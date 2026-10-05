# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE = 0.04


def bs_d1(S, K, T, sigma, r=RISK_FREE_RATE):
    sigma = max(float(sigma), 1e-6)
    T = max(float(T), 1e-6)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S, K, T, sigma, r=RISK_FREE_RATE):
    return bs_d1(S, K, T, sigma, r) - sigma * np.sqrt(T)


def bs_delta(S, K, T, sigma, option_type):
    d1 = bs_d1(S, K, T, sigma)
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1


def bs_gamma(S, K, T, sigma):
    d1 = bs_d1(S, K, T, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, sigma):
    d1 = bs_d1(S, K, T, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def bs_theta(S, K, T, sigma, option_type, r=RISK_FREE_RATE):
    d1 = bs_d1(S, K, T, sigma)
    d2 = bs_d2(S, K, T, sigma)

    first = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        second = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        second = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return (first + second) / 252


def bs_vanna(S, K, T, sigma):
    d1 = bs_d1(S, K, T, sigma)
    d2 = bs_d2(S, K, T, sigma)
    return -norm.pdf(d1) * d2 / sigma


def bs_volga(S, K, T, sigma):
    d1 = bs_d1(S, K, T, sigma)
    d2 = bs_d2(S, K, T, sigma)
    raw_vega = S * norm.pdf(d1) * np.sqrt(T)
    return raw_vega * d1 * d2 / sigma


def bs_charm(S, K, T, sigma, option_type, r=RISK_FREE_RATE):
    d1 = bs_d1(S, K, T, sigma, r)
    d2 = bs_d2(S, K, T, sigma, r)

    numerator = 2 * r * T - d2 * sigma * np.sqrt(T)
    raw_charm = -norm.pdf(d1) * numerator / (2 * T * sigma * np.sqrt(T))

    return raw_charm / 252


def compute_spread_greeks_from_mark(row):
    S = float(row["underlying_price_now"])
    T = max(float(row["dte_remaining"]) / 252, 1 / 252)

    strategy = row["strategy"]
    option_type = "put" if strategy == "put_credit_spread" else "call"

    short_k = float(row["short_strike"])
    long_k = float(row["long_strike"])

    short_iv = float(row["short_iv"])
    long_iv = float(row["long_iv"])

    short = {
        "delta": bs_delta(S, short_k, T, short_iv, option_type),
        "gamma": bs_gamma(S, short_k, T, short_iv),
        "vega": bs_vega(S, short_k, T, short_iv),
        "theta": bs_theta(S, short_k, T, short_iv, option_type),
        "vanna": bs_vanna(S, short_k, T, short_iv),
        "volga": bs_volga(S, short_k, T, short_iv),
        "charm": bs_charm(S, short_k, T, short_iv, option_type),
    }

    long = {
        "delta": bs_delta(S, long_k, T, long_iv, option_type),
        "gamma": bs_gamma(S, long_k, T, long_iv),
        "vega": bs_vega(S, long_k, T, long_iv),
        "theta": bs_theta(S, long_k, T, long_iv, option_type),
        "vanna": bs_vanna(S, long_k, T, long_iv),
        "volga": bs_volga(S, long_k, T, long_iv),
        "charm": bs_charm(S, long_k, T, long_iv, option_type),
    }

    return {
        f"spread_{g}": -short[g] + long[g]
        for g in ["delta", "gamma", "vega", "theta", "vanna", "volga", "charm"]
    }
