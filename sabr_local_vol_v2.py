# region imports
from AlgorithmImports import *
# endregion


"""
SABR and local-vol utilities for path simulation and option repricing.

This is a practical sticky-SABR/local-vol implementation:
- Hagan lognormal SABR implied volatility.
- Local volatility approximated from the SABR smile slope/curvature.
- Black-Scholes is used only as the pricing transform after SABR supplies the
  strike- and state-dependent implied volatility.

The module does not read or write files.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


EPS = 1e-12


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def hagan_lognormal_iv(
    forward: np.ndarray | float,
    strike: np.ndarray | float,
    maturity: np.ndarray | float,
    params: SABRParams,
) -> np.ndarray:
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(maturity, dtype=float)

    f = np.maximum(f, EPS)
    k = np.maximum(k, EPS)
    t = np.maximum(t, EPS)

    alpha = max(float(params.alpha), EPS)
    beta = float(np.clip(params.beta, 0.0, 1.0))
    rho = float(np.clip(params.rho, -0.999, 0.999))
    nu = max(float(params.nu), EPS)

    one_minus_beta = 1.0 - beta
    log_fk = np.log(f / k)
    fk_beta = (f * k) ** (0.5 * one_minus_beta)

    z = (nu / alpha) * fk_beta * log_fk
    x_z = np.log(
        (np.sqrt(np.maximum(1.0 - 2.0 * rho * z + z * z, EPS)) + z - rho)
        / (1.0 - rho)
    )
    z_over_x = np.where(np.abs(z) < 1e-8, 1.0, z / np.maximum(x_z, EPS))

    denominator = fk_beta * (
        1.0
        + (one_minus_beta**2 / 24.0) * log_fk**2
        + (one_minus_beta**4 / 1920.0) * log_fk**4
    )

    correction = (
        (one_minus_beta**2 / 24.0) * alpha**2 / np.maximum((f * k) ** one_minus_beta, EPS)
        + 0.25 * rho * beta * nu * alpha / np.maximum(fk_beta, EPS)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )

    iv = alpha / np.maximum(denominator, EPS) * z_over_x * (1.0 + correction * t)

    atm = np.isclose(f, k, rtol=0.0, atol=1e-8 * np.maximum(f, 1.0))
    if np.any(atm):
        f_beta = f ** one_minus_beta
        atm_iv = alpha / np.maximum(f_beta, EPS) * (
            1.0
            + (
                (one_minus_beta**2 / 24.0) * alpha**2 / np.maximum(f ** (2.0 * one_minus_beta), EPS)
                + 0.25 * rho * beta * nu * alpha / np.maximum(f_beta, EPS)
                + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
            )
            * t
        )
        iv = np.where(atm, atm_iv, iv)

    return np.clip(iv, 0.01, 5.00)


def approximate_local_vol(
    spot: np.ndarray | float,
    maturity: np.ndarray | float,
    params: SABRParams,
    reference_forward: float,
) -> np.ndarray:
    """
    Stable local-vol approximation derived from the SABR smile.

    A full Dupire implementation requires a dense, arbitrage-clean call surface.
    This approximation uses the SABR ATM level and finite-difference smile
    derivatives, while preserving positivity and avoiding unstable denominator
    explosions.
    """
    s = np.asarray(spot, dtype=float)
    t = np.maximum(np.asarray(maturity, dtype=float), 1.0 / 252.0)
    f = np.maximum(s, EPS)

    bump = 0.01
    k_mid = np.maximum(f, EPS)
    k_down = np.maximum(k_mid * np.exp(-bump), EPS)
    k_up = k_mid * np.exp(bump)

    iv_mid = hagan_lognormal_iv(f, k_mid, t, params)
    iv_down = hagan_lognormal_iv(f, k_down, t, params)
    iv_up = hagan_lognormal_iv(f, k_up, t, params)

    slope = (iv_up - iv_down) / (2.0 * bump)
    curvature = (iv_up - 2.0 * iv_mid + iv_down) / (bump**2)

    log_displacement = np.log(np.maximum(s, EPS) / max(reference_forward, EPS))
    local = iv_mid * (
        1.0
        + 0.35 * slope * log_displacement
        + 0.10 * curvature * log_displacement**2
    )
    return np.clip(local, 0.01, 5.00)


def black_scholes_price(
    spot: np.ndarray | float,
    strike: float,
    maturity: np.ndarray | float,
    rate: float,
    volatility: np.ndarray | float,
    option_type: str,
) -> np.ndarray:
    s = np.maximum(np.asarray(spot, dtype=float), EPS)
    k = max(float(strike), EPS)
    t = np.maximum(np.asarray(maturity, dtype=float), 0.0)
    sigma = np.maximum(np.asarray(volatility, dtype=float), EPS)

    intrinsic = (
        np.maximum(s - k, 0.0)
        if str(option_type).lower().startswith("c")
        else np.maximum(k - s, 0.0)
    )

    active = t > 1e-10
    sqrt_t = np.sqrt(np.maximum(t, EPS))
    d1 = (np.log(s / k) + (rate + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if str(option_type).lower().startswith("c"):
        value = s * ndtr(d1) - k * np.exp(-rate * t) * ndtr(d2)
    else:
        value = k * np.exp(-rate * t) * ndtr(-d2) - s * ndtr(-d1)

    return np.where(active, value, intrinsic)


def sticky_sabr_option_price(
    spot: np.ndarray | float,
    strike: float,
    maturity: np.ndarray | float,
    rate: float,
    option_type: str,
    params: SABRParams,
    atm_vol_multiplier: np.ndarray | float = 1.0,
) -> np.ndarray:
    """
    Reprice an option using the SABR smile at the current simulated spot.

    The SABR alpha parameter is scaled by atm_vol_multiplier to represent
    regime-dependent ATM-vol state evolution.
    """
    multiplier = np.maximum(np.asarray(atm_vol_multiplier, dtype=float), 0.25)
    effective = SABRParams(
        alpha=float(params.alpha),
        beta=float(params.beta),
        rho=float(params.rho),
        nu=float(params.nu),
    )

    forward = np.asarray(spot, dtype=float) * np.exp(rate * np.asarray(maturity, dtype=float))
    base_iv = hagan_lognormal_iv(forward, strike, maturity, effective)
    iv = np.clip(base_iv * multiplier, 0.01, 5.00)
    return black_scholes_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=iv,
        option_type=option_type,
    )
