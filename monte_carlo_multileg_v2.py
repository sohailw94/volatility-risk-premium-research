# region imports
from AlgorithmImports import *
# endregion

"""
Regime-aware Monte Carlo with:
- HMM regime probabilities;
- sticky-SABR/local-vol spot dynamics;
- pathwise SABR option repricing;
- multi-leg entry/exit slippage;
- aligned joint portfolio scenarios.

Run:
    python -m src.models.monte_carlo_multileg
"""


import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import qc_config_v2 as cfg
from execution_slippage_v2 import estimate_strategy_slippage
from sabr_local_vol_v2 import (
    SABRParams,
    approximate_local_vol,
    sticky_sabr_option_price,
)


N_PATHS = cfg.N_PATHS
MAX_DAYS = cfg.MC_MAX_DAYS
TRADING_DAYS = cfg.TRADING_DAYS
CONTRACT_MULTIPLIER = cfg.CONTRACT_MULTIPLIER
RISK_FREE_RATE = cfg.RISK_FREE_RATE

VOL_MULTIPLIERS = cfg.VOL_MULTIPLIERS
BASE_STUDENT_T_DF = 6.0

# Pathwise exit assumptions. Replace with your exact exit engine thresholds if
# they differ.
PROFIT_TARGET_FRACTION = cfg.PROFIT_TARGET_FRACTION
STOP_LOSS_MULTIPLE = cfg.STOP_LOSS_MULTIPLE
MIN_EXIT_DTE = cfg.MIN_EXIT_DTE


def _safe_float(value, default=np.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def normalize_option_type(value):
    return "call" if str(value).strip().lower().startswith("c") else "put"


def parse_legs(row):
    value = row["legs_json"]
    return value if isinstance(value, list) else json.loads(value)


def signed_quantity(leg):
    signed = int(leg.get("signed_qty", 0) or 0)
    if signed:
        return signed
    quantity = int(leg.get("qty", 1) or 1)
    return quantity if str(leg.get("side", "")).upper() == "BUY" else -quantity


def stable_trade_id(row):
    raw = "|".join(
        [
            str(row.get("snapshot_date", row.get("date", ""))),
            str(row.get("ticker", "")),
            str(row.get("structure", "")),
            str(row.get("expiry", "")),
            str(row.get("legs_json", "")),
        ]
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def deterministic_seed(*parts):
    raw = "|".join(str(part) for part in parts).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def ensure_regime_defaults(frame):
    frame=frame.copy()
    defaults={
        "hmm_regime":"NORMAL","hmm_prob_calm":0.0,"hmm_prob_normal":1.0,"hmm_prob_stress":0.0,
        "hmm_jump_multiplier":1.0,"hmm_factor_loading":0.40,"hmm_slippage_multiplier":1.0,
    }
    for c,v in defaults.items():
        if c not in frame.columns:frame[c]=v
        else:frame[c]=frame[c].fillna(v)
    return frame


def sabr_params_from_row(row):
    return SABRParams(
        alpha=max(_safe_float(row.get("sabr_alpha"), 0.30), 0.01),
        beta=float(np.clip(_safe_float(row.get("sabr_beta"), 0.60), 0.0, 1.0)),
        rho=float(np.clip(_safe_float(row.get("sabr_rho"), -0.20), -0.999, 0.999)),
        nu=max(_safe_float(row.get("sabr_nu"), 0.50), 0.01),
    )


def entry_strategy_value_per_share(row):
    premium_type = str(row.get("premium_type", "")).upper()
    net_credit = _safe_float(row.get("net_credit"), 0.0)
    net_debit = _safe_float(row.get("net_debit"), 0.0)

    if premium_type == "CREDIT":
        return -abs(net_credit)
    return abs(net_debit)



def modeled_executable_entry(row):
    """Use the same executable package basis that order submission uses.

    filter_liquid_strategies stores ``entry_execution_signed`` after crossing
    the displayed natural by the configured marketable cushion.  Using that
    basis here removes the old midpoint/slippage mismatch between MC/HJB and
    live execution.  The legacy slippage proxy remains only as a compatibility
    fallback for rows created outside the current workflow.
    """
    premium_type=str(row.get("premium_type","")).upper()
    theoretical_credit=abs(_safe_float(row.get("net_credit"),0.0))
    theoretical_debit=abs(_safe_float(row.get("net_debit"),0.0))
    signed_exec=_safe_float(row.get("entry_execution_signed"),np.nan)

    if np.isfinite(signed_exec):
        if premium_type=="CREDIT" and signed_exec<0.0:
            executable=max(-signed_exec,1e-6)
            slip=max(theoretical_credit-executable,0.0)*CONTRACT_MULTIPLIER
            return premium_type,theoretical_credit,executable,slip
        if premium_type!="CREDIT" and signed_exec>0.0:
            executable=max(signed_exec,1e-6)
            slip=max(executable-theoretical_debit,0.0)*CONTRACT_MULTIPLIER
            return premium_type,theoretical_debit,executable,slip

    entry_slippage=float(estimate_strategy_slippage(row,contracts=1,phase="ENTRY"))
    slippage_per_share=entry_slippage/CONTRACT_MULTIPLIER
    if premium_type=="CREDIT":
        return premium_type,theoretical_credit,max(theoretical_credit-slippage_per_share,1e-6),entry_slippage
    return premium_type,theoretical_debit,max(theoretical_debit+slippage_per_share,1e-6),entry_slippage

def strategy_mark_per_share(row, spot, maturity, vol_multiplier):
    params = sabr_params_from_row(row)
    mark = np.zeros_like(np.asarray(spot, dtype=float))

    for leg in parse_legs(row):
        price = sticky_sabr_option_price(
            spot=spot,
            strike=float(leg["strike"]),
            maturity=maturity,
            rate=RISK_FREE_RATE,
            option_type=normalize_option_type(leg["option_type"]),
            params=params,
            atm_vol_multiplier=vol_multiplier,
        )
        mark += signed_quantity(leg) * price

    return mark



def entry_leg_prices_and_basis(row):
    prices=[]; basis=[]
    for leg in parse_legs(row):
        q=signed_quantity(leg)
        bid=_safe_float(leg.get("bid"),np.nan)
        ask=_safe_float(leg.get("ask"),np.nan)
        mid=_safe_float(leg.get("mid"),np.nan)
        px=ask if q>0 else bid
        if not np.isfinite(px) or px<=0.0:
            px=max(mid,1e-6)
        prices.append(float(px))
        basis.append(abs(q)*float(px)*CONTRACT_MULTIPLIER)
    return np.asarray(prices,float),np.asarray(basis,float)


def leg_marks_per_share(row, spot, maturity, vol_multiplier):
    params=sabr_params_from_row(row)
    values=[]
    for leg in parse_legs(row):
        values.append(sticky_sabr_option_price(
            spot=spot,strike=float(leg["strike"]),maturity=maturity,
            rate=RISK_FREE_RATE,option_type=normalize_option_type(leg["option_type"]),
            params=params,atm_vol_multiplier=vol_multiplier,
        ))
    return np.column_stack(values) if values else np.empty((len(np.asarray(spot)),0))

def standardized_t(rng, shape, degrees_freedom):
    draws = student_t.rvs(
        df=degrees_freedom,
        size=shape,
        random_state=rng,
    )
    return draws / np.sqrt(degrees_freedom / (degrees_freedom - 2.0))


def regime_degrees_freedom(row):
    return float(
        row.get("hmm_prob_calm", 0.0) * getattr(cfg, "MC_DF_CALM", 10.0)
        + row.get("hmm_prob_normal", 1.0) * getattr(cfg, "MC_DF_NORMAL", 6.0)
        + row.get("hmm_prob_stress", 0.0) * getattr(cfg, "MC_DF_STRESS", 4.0)
    )


def simulate_spot_and_vol_paths(row, sigma_multiplier, scenario_label):
    snapshot = pd.Timestamp(row["snapshot_date"]).normalize()
    expiry = pd.Timestamp(row["expiry"]).normalize()
    dte = int((expiry - snapshot).days)
    days = int(min(max(dte, 1), MAX_DAYS))
    dt = 1.0 / TRADING_DAYS

    ticker = str(row["ticker"])
    trade_id = stable_trade_id(row)

    market_rng = np.random.default_rng(
        deterministic_seed(snapshot.date(), scenario_label, "MARKET")
    )
    idio_rng = np.random.default_rng(
        deterministic_seed(snapshot.date(), scenario_label, ticker, trade_id)
    )

    df = max(regime_degrees_freedom(row), 3.1)
    market = standardized_t(market_rng, (N_PATHS, days), df)
    idio = standardized_t(idio_rng, (N_PATHS, days), df)

    loading = float(np.clip(row.get("hmm_factor_loading", 0.40), 0.0, 0.95))
    shocks = loading * market + np.sqrt(max(1.0 - loading**2, 0.0)) * idio

    spot0 = float(row["underlying_price"])
    params = sabr_params_from_row(row)
    spots = np.empty((N_PATHS, days + 1), dtype=float)
    vol_state = np.empty((N_PATHS, days + 1), dtype=float)
    spots[:, 0] = spot0
    vol_state[:, 0] = sigma_multiplier

    jump_multiplier = float(row.get("hmm_jump_multiplier", 1.0))
    jump_probability = (
        jump_multiplier
        * getattr(cfg, "MC_JUMP_FREQUENCY_SCALE", 1.0)
        / TRADING_DAYS
    )

    for day in range(days):
        remaining = max((days - day) / TRADING_DAYS, 1.0 / TRADING_DAYS)
        local_vol = approximate_local_vol(
            spot=spots[:, day],
            maturity=remaining,
            params=params,
            reference_forward=spot0,
        )
        local_vol *= (
            vol_state[:, day]
            * getattr(cfg, "MC_PATH_VOL_SCALE", 1.0)
        )

        jump_occurs = idio_rng.random(N_PATHS) < jump_probability
        jump_sign = idio_rng.choice(np.array([-1.0, 1.0]), size=N_PATHS)
        jump_size = np.abs(
            idio_rng.normal(
                loc=getattr(cfg, "MC_JUMP_LOC_DAILY_SIGMA", 2.0)
                * local_vol / np.sqrt(TRADING_DAYS),
                scale=getattr(cfg, "MC_JUMP_SCALE_DAILY_SIGMA", 0.5)
                * local_vol / np.sqrt(TRADING_DAYS),
                size=N_PATHS,
            )
        )

        log_return = (
            -0.5 * local_vol**2 * dt
            + local_vol * np.sqrt(dt) * shocks[:, day]
            + jump_occurs * jump_sign * jump_size
        )
        spots[:, day + 1] = spots[:, day] * np.exp(log_return)

        # Mean-reverting ATM-vol multiplier with regime-sensitive vol-of-vol.
        stress = float(row.get("hmm_prob_stress", 0.0))
        vol_of_vol = (
            getattr(cfg, "MC_VOL_OF_VOL_BASE", 0.12)
            + getattr(cfg, "MC_VOL_OF_VOL_STRESS", 0.25) * stress
        )
        reversion = getattr(cfg, "MC_VOL_STATE_REVERSION", 0.10)
        vol_state[:, day + 1] = np.clip(
            vol_state[:, day]
            + reversion * (sigma_multiplier - vol_state[:, day]) * dt
            + vol_of_vol * np.sqrt(dt) * idio_rng.normal(size=N_PATHS),
            0.35,
            2.50,
        )

    return spots, vol_state, dte


def pathwise_strategy_pnl(row, spots, vol_state, dte):
    paths=spots.shape[0]
    days=spots.shape[1]-1
    premium_type,theoretical_entry_premium,executable_entry_premium,entry_slippage=modeled_executable_entry(row)
    normal_exit_slippage=estimate_strategy_slippage(row,contracts=1,phase="EXIT")
    stop_exit_slippage=estimate_strategy_slippage(row,contracts=1,phase="STOP_EXIT")

    pnl=np.full(paths,np.nan)
    exit_day=np.full(paths,days,dtype=int)
    stopped=np.zeros(paths,dtype=bool)
    active=np.ones(paths,dtype=bool)

    entry_prices,leg_basis=entry_leg_prices_and_basis(row)
    gross_basis=max(float(np.sum(leg_basis)),1e-9)
    management_per_share=_safe_float(row.get("management_risk_basis_per_share",row.get("max_loss")),0.0)
    risk_basis=max(management_per_share*CONTRACT_MULTIPLIER,gross_basis,1e-9)
    largest_leg_basis=max(float(np.max(leg_basis)) if leg_basis.size else 0.0,1e-9)
    package_only=bool(row.get("package_only_management",False))
    leg_q=np.asarray([signed_quantity(x) for x in parse_legs(row)],float)

    strategy_floor=np.full(paths,np.nan)
    strategy_next=np.full(paths,max(
        float(getattr(cfg,"STRATEGY_TRAIL_START_PCT",0.15))*risk_basis,
        float(getattr(cfg,"LEG_TRAIL_START_PCT",0.25))*largest_leg_basis,
    ))
    leg_floor=np.full((paths,len(leg_q)),np.nan)
    leg_next=np.full((paths,len(leg_q)),float(getattr(cfg,"LEG_TRAIL_START_PCT",0.25)))

    strategy_stop=float(getattr(cfg,"STRATEGY_STOP_TRIGGER_PCT",0.08))
    leg_stop=float(getattr(cfg,"LEG_STOP_LOSS_PCT",0.15))
    trail_step=float(getattr(cfg,"TRAIL_STEP_PCT",0.10))
    min_exit_dte=int(row.get('min_exit_dte',MIN_EXIT_DTE) if pd.notna(row.get('min_exit_dte',MIN_EXIT_DTE)) else MIN_EXIT_DTE)

    entry_credit=executable_entry_premium if premium_type=="CREDIT" else 0.0
    entry_debit=executable_entry_premium if premium_type!="CREDIT" else 0.0

    for day in range(1,days+1):
        if not active.any():
            break
        remaining_days=max(dte-day,0)
        maturity=remaining_days/TRADING_DAYS
        active_idx=np.flatnonzero(active)

        leg_marks=leg_marks_per_share(
            row=row,spot=spots[active_idx,day],maturity=maturity,
            vol_multiplier=vol_state[active_idx,day],
        )
        signed_mark=leg_marks @ leg_q
        if premium_type=="CREDIT":
            gross=(signed_mark+entry_credit)*CONTRACT_MULTIPLIER
        else:
            gross=(signed_mark-entry_debit)*CONTRACT_MULTIPLIER

        leg_pnl=(leg_marks-entry_prices.reshape(1,-1))*leg_q.reshape(1,-1)*CONTRACT_MULTIPLIER
        leg_ret=leg_pnl/np.maximum(leg_basis.reshape(1,-1),1e-9)

        # Structure stop: trigger at -8% of gross premium basis.  Production
        # execution then works its limit ladder and hard-flattens if deterioration
        # reaches the -10% hard-stop threshold.
        stop_local=(gross/risk_basis)<=-strategy_stop
        leg_stop_local=np.zeros(len(active_idx),dtype=bool) if package_only else np.any(leg_ret<=-leg_stop,axis=1)

        sf=strategy_floor[active_idx].copy()
        sn=strategy_next[active_idx].copy()
        cross=gross>=sn
        inc=np.where(cross,np.floor(np.maximum(gross-sn,0.0)/(trail_step*risk_basis)).astype(int)+1,0)
        old_sf=sf.copy()
        sf=np.where(cross,sn+(inc-1)*trail_step*risk_basis,sf)
        sn=np.where(cross,sn+inc*trail_step*risk_basis,sn)
        trail_local=np.isfinite(old_sf)&(~cross)&(gross<=old_sf)
        strategy_floor[active_idx]=sf; strategy_next[active_idx]=sn

        lf=leg_floor[active_idx].copy()
        ln=leg_next[active_idx].copy()
        if package_only:
            leg_trail_local=np.zeros(len(active_idx),dtype=bool)
        else:
            lcross=leg_ret>=ln
            linc=np.where(lcross,np.floor(np.maximum(leg_ret-ln,0.0)/trail_step).astype(int)+1,0)
            old_lf=lf.copy()
            lf=np.where(lcross,ln+(linc-1)*trail_step,lf)
            ln=np.where(lcross,ln+linc*trail_step,ln)
            leg_trail_local=np.any(np.isfinite(old_lf)&(~lcross)&(leg_ret<=old_lf),axis=1)
            leg_floor[active_idx]=lf; leg_next[active_idx]=ln

        forced_dte=remaining_days<=min_exit_dte
        stop_any=stop_local|leg_stop_local
        normal_any=trail_local|leg_trail_local
        if forced_dte:
            normal_any=~stop_any

        normal_idx=active_idx[normal_any & ~stop_any]
        stop_idx=active_idx[stop_any]

        pnl[normal_idx]=gross[normal_any & ~stop_any]-normal_exit_slippage
        pnl[stop_idx]=gross[stop_any]-stop_exit_slippage
        exited=np.concatenate((normal_idx,stop_idx))
        if exited.size:
            exit_day[exited]=day
            stopped[stop_idx]=True
            active[exited]=False

    if active.any():
        active_idx=np.flatnonzero(active)
        marks=strategy_mark_per_share(
            row=row,spot=spots[active_idx,-1],maturity=0.0,
            vol_multiplier=vol_state[active_idx,-1],
        )
        gross=(marks+entry_credit)*CONTRACT_MULTIPLIER if premium_type=="CREDIT" else (marks-entry_debit)*CONTRACT_MULTIPLIER
        pnl[active_idx]=gross-normal_exit_slippage

    return pnl,exit_day,stopped


def summarize(pnl, exit_day, stopped):
    q05 = float(np.quantile(pnl, 0.05))
    tail = pnl[pnl <= q05]
    std = float(np.std(pnl))

    return {
        "mc_expected_pnl": float(np.mean(pnl)),
        "mc_median_pnl": float(np.median(pnl)),
        "mc_pop": float(np.mean(pnl > 0.0)),
        "mc_cvar_95": float(abs(np.mean(tail))) if tail.size else np.nan,
        "mc_expected_holding_days": float(np.mean(exit_day)),
        "mc_stop_probability": float(np.mean(stopped)),
        "mc_expected_sharpe": float(np.mean(pnl) / std) if std > 0.0 else 0.0,
    }


def run_monte_carlo(frame: pd.DataFrame, n_paths: int | None = None):
    global N_PATHS
    N_PATHS=int(n_paths or cfg.N_PATHS)
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame({"scenario_id":np.arange(N_PATHS,dtype=np.int32)})
    frame=frame.copy()
    frame["snapshot_date"]=pd.to_datetime(frame.get("snapshot_date",frame.get("date")),errors="coerce").dt.normalize()
    frame["expiry"]=pd.to_datetime(frame["expiry"],errors="coerce").dt.normalize()
    frame["dte_recomputed"]=(frame["expiry"]-frame["snapshot_date"]).dt.days
    frame=frame[frame["dte_recomputed"]>0].copy()
    frame=ensure_regime_defaults(frame)
    output_rows=[]; scenario_columns={"scenario_id":np.arange(N_PATHS,dtype=np.int32)}
    for _,row in frame.iterrows():
        summaries=[]; base_pnl=None; base_summary=None
        for multiplier in VOL_MULTIPLIERS:
            label=f"VOL_{multiplier:.2f}"
            spots,vol_state,dte=simulate_spot_and_vol_paths(row,multiplier,label)
            pnl,exit_day,stopped=pathwise_strategy_pnl(row,spots,vol_state,dte)
            summary=summarize(pnl,exit_day,stopped); summary["scenario_label"]=label; summaries.append(summary)
            if np.isclose(multiplier,1.0):base_pnl=pnl;base_summary=summary
        expected_values=np.array([item["mc_expected_pnl"] for item in summaries]); cvars=np.array([item["mc_cvar_95"] for item in summaries])
        result=row.to_dict(); result.update(base_summary); result.update({
            "mc_worst_case_expected_pnl":float(expected_values.min()),
            "mc_p25_scenario_expected_pnl":float(np.quantile(expected_values,0.25)),
            "mc_worst_case_cvar_95":float(cvars.max()),
            "mc_positive_ev_scenario_fraction":float(np.mean(expected_values>0.0)),
            "mc_scenario_count":len(summaries),
            "mc_model":"HMM_STICKY_SABR_LOCAL_VOL_ASYM_V5_PACKAGE_AWARE_EXITS",
            "mc_trade_id":stable_trade_id(row),
            "mc_factor_loading":float(row.get("hmm_factor_loading",0.40)),
            "mc_entry_slippage":estimate_strategy_slippage(row,1,"ENTRY"),
            "mc_exit_slippage":estimate_strategy_slippage(row,1,"EXIT"),
            "mc_stop_exit_slippage":estimate_strategy_slippage(row,1,"STOP_EXIT"),
            "mc_theoretical_entry_premium":modeled_executable_entry(row)[1],
            "mc_executable_entry_premium":modeled_executable_entry(row)[2],
        })
        output_rows.append(result); scenario_columns[result["mc_trade_id"]]=base_pnl
    return pd.DataFrame(output_rows),pd.DataFrame(scenario_columns)

