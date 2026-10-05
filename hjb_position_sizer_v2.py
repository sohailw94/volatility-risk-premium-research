from AlgorithmImports import *
# endregion


import numpy as np
import pandas as pd
import qc_config_v2 as cfg

DEFAULT_EQUITY = float(cfg.START_CASH)
ACTION_GRID = np.array(cfg.ACTION_GRID, dtype=float)
POSITION_PCT_PER_ACTION = cfg.POSITION_PCT_PER_ACTION

MAX_TOTAL_DEFINED_RISK_PCT = cfg.MAX_TOTAL_DEFINED_RISK_PCT
MAX_SINGLE_TRADE_RISK_PCT = cfg.MAX_SINGLE_TRADE_RISK_PCT
MAX_TICKER_RISK_PCT = cfg.MAX_TICKER_RISK_PCT
UNCAPPED_MAX_SINGLE_TRADE_RISK_PCT = float(getattr(cfg,'UNCAPPED_MAX_SINGLE_TRADE_RISK_PCT',MAX_SINGLE_TRADE_RISK_PCT))
UNCAPPED_MAX_TICKER_RISK_PCT = float(getattr(cfg,'UNCAPPED_MAX_TICKER_RISK_PCT',MAX_TICKER_RISK_PCT))
UNCAPPED_POSITION_PCT_PER_ACTION = float(getattr(cfg,'UNCAPPED_POSITION_PCT_PER_ACTION',POSITION_PCT_PER_ACTION))

RANDOM_SEED = cfg.HJB_RANDOM_SEED
N_RANDOM_STARTS = cfg.HJB_RANDOM_STARTS
MAX_PASSES = cfg.HJB_MAX_PASSES


def row_float(row, key, default):
    x=sf(row.get(key,default),default)
    return float(default) if not np.isfinite(x) else float(x)


def sf(x, default=np.nan):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def decision(a):
    if a <= 0:
        return "NO_TRADE"
    if a <= 0.25:
        return "TRADE_TINY"
    if a <= 0.50:
        return "TRADE_SMALL"
    return "TRADE_NORMAL"


def risk_full(row):
    x = sf(row.get("max_loss"))
    if np.isfinite(x) and x > 0:
        return 100.0 * x if x < 100.0 else x
    x = sf(row.get("net_debit"))
    if np.isfinite(x) and x > 0:
        return 100.0 * x
    x = sf(row.get("mc_cvar_95"), 0.0)
    return max(x, 0.0)


def portfolio_state(algorithm):
    equity=max(float(algorithm.portfolio.total_portfolio_value),1.0)
    total_risk=0.0;ticker_risk={}
    for p in getattr(algorithm,"_vh_open_structures",{}).values():
        if p.get("closed"):continue
        rd=max(sf(p.get("risk_dollars"),0.0),0.0);total_risk+=rd;t=str(p.get("ticker","")).upper();ticker_risk[t]=ticker_risk.get(t,0.0)+100.0*rd/equity
    return equity,100.0*total_risk/equity,ticker_risk


def allowed_actions_for_row(row):
    if str(row.get("risk_class","CAPPED")).upper()=="UNCAPPED":
        return np.array([0.0,1.0],dtype=float)
    return ACTION_GRID


def nearest_grid(x, grid=None):
    g=ACTION_GRID if grid is None else np.asarray(grid,float)
    return float(g[np.argmin(np.abs(g - sf(x, 0.0)))])


class Problem:
    def __init__(self, trades, indices, pnl, equity, current_risk_pct, ticker_risk_pct):
        self.trades = trades
        self.indices = indices
        self.pnl = pnl
        self.n = len(indices)
        self.equity = equity
        self.risk = np.array([risk_full(trades.loc[i]) for i in indices], float)
        self.ticker = np.array([str(trades.loc[i].get("ticker", "")).upper() for i in indices], object)
        self.uncapped = np.array([str(trades.loc[i].get("risk_class","CAPPED")).upper()=="UNCAPPED" for i in indices], bool)
        self.allowed = [allowed_actions_for_row(trades.loc[i]) for i in indices]
        self.single_limit_pct = np.array([
            row_float(
                trades.loc[i],
                "hjb_max_single_risk_pct",
                UNCAPPED_MAX_SINGLE_TRADE_RISK_PCT if self.uncapped[j] else MAX_SINGLE_TRADE_RISK_PCT,
            )
            for j,i in enumerate(indices)
        ],float)
        self.ticker_limit_pct = np.array([
            row_float(
                trades.loc[i],
                "hjb_max_ticker_risk_pct",
                UNCAPPED_MAX_TICKER_RISK_PCT if self.uncapped[j] else MAX_TICKER_RISK_PCT,
            )
            for j,i in enumerate(indices)
        ],float)
        self.current_risk_pct = current_risk_pct
        self.ticker_risk_pct = ticker_risk_pct
        self.available_risk = max(MAX_TOTAL_DEFINED_RISK_PCT-current_risk_pct, 0.0)/100.0*equity

    def wealth(self, a):
        # IMPORTANT: mc_joint_scenario_pnl is already net of MC entry/exit slippage.
        return self.equity + self.pnl @ a

    def utility(self, a):
        w = self.wealth(a)
        return -np.inf if np.any(w <= 0) else float(np.mean(np.log(w)))

    def feasible(self, a):
        if float(self.risk @ a) > self.available_risk + 1e-9:
            return False
        if np.any(self.risk*a > self.single_limit_pct/100.0*self.equity + 1e-9):
            return False
        for t in set(self.ticker):
            p = np.where(self.ticker == t)[0]
            new = float(self.risk[p] @ a[p])
            old = self.ticker_risk_pct.get(str(t), 0.0)/100.0*self.equity
            active=p[a[p]>0]
            cap_pct=float(np.min(self.ticker_limit_pct[active])) if len(active) else float(np.max(self.ticker_limit_pct[p]))
            if old + new > cap_pct/100.0*self.equity + 1e-9:
                return False
        return bool(np.all(self.wealth(a) > 0))


def load_problem(trades, scenarios, equity, current_risk_pct, ticker_risk_pct):
    sc=scenarios.copy();indices=[];cols=[]
    for i,r in trades.iterrows():
        tid=str(r.get("mc_trade_id",""));rf=risk_full(r)
        if tid in sc.columns and np.isfinite(rf) and rf>0:
            v=pd.to_numeric(sc[tid],errors="coerce")
            if v.notna().all():indices.append(i);cols.append(tid)
    matrix=sc[cols].to_numpy(float) if cols else np.empty((len(sc),0))
    return Problem(trades,indices,matrix,equity,current_risk_pct,ticker_risk_pct)


def repair(problem, a):
    a = np.array([nearest_grid(x,problem.allowed[j]) for j,x in enumerate(a)], float)
    while not problem.feasible(a) and np.any(a > 0):
        best = None
        u0 = problem.utility(a)
        for j in np.where(a > 0)[0]:
            lower = problem.allowed[j][problem.allowed[j] < a[j]-1e-12]
            if not len(lower):
                continue
            trial = a.copy()
            trial[j] = lower[-1]
            u = problem.utility(trial)
            loss = (u0-u) if np.isfinite(u0) and np.isfinite(u) else 0.0
            if best is None or loss < best[0]:
                best = (loss, trial)
        if best is None:
            return np.zeros(problem.n)
        a = best[1]
    return a


def coordinate_ascent(problem, start, rng):
    a = repair(problem, start)
    u = problem.utility(a)
    for passes in range(1, MAX_PASSES+1):
        changed = False
        for j in rng.permutation(problem.n):
            bj, bu = a[j], u
            for x in problem.allowed[j]:
                if np.isclose(x, a[j]):
                    continue
                trial = a.copy()
                trial[j] = x
                if not problem.feasible(trial):
                    continue
                ut = problem.utility(trial)
                if ut > bu + 1e-12:
                    bj, bu = float(x), ut
            if not np.isclose(bj, a[j]):
                a[j], u, changed = bj, bu, True
        if not changed:
            return a, u, passes
    return a, u, MAX_PASSES


def greedy(problem):
    a = np.zeros(problem.n)
    u = problem.utility(a)
    while True:
        choice = None
        for j in range(problem.n):
            for x in problem.allowed[j]:
                if x <= a[j]+1e-12:
                    continue
                trial = a.copy()
                trial[j] = x
                if not problem.feasible(trial):
                    continue
                ut = problem.utility(trial)
                if ut > u + 1e-12 and (choice is None or ut > choice[0]):
                    choice = (ut, trial)
        if choice is None:
            return a
        u, a = choice


def optimize(problem, trades):
    rng = np.random.default_rng(RANDOM_SEED)
    starts = [("CASH", np.zeros(problem.n)), ("GREEDY", greedy(problem))]

    wealth_start = np.array(
        [nearest_grid(trades.loc[i].get("wealth_action_size", 0.0),problem.allowed[j]) for j,i in enumerate(problem.indices)],
        float,
    )
    starts.append(("WEALTH", repair(problem, wealth_start)))

    # Explicit singleton starts jump over the zero/fixed-cost discontinuity.
    for j in range(problem.n):
        for x in problem.allowed[j][problem.allowed[j]>0]:
            a = np.zeros(problem.n)
            a[j] = x
            if problem.feasible(a):
                starts.append((f"SINGLE_{j}_{x}", a))

    for k in range(N_RANDOM_STARTS):
        a=np.zeros(problem.n)
        for j in range(problem.n):
            if problem.uncapped[j]:
                a[j]=rng.choice(problem.allowed[j],p=[0.80,0.20])
            else:
                a[j]=rng.choice(problem.allowed[j],p=[0.55,0.20,0.15,0.10])
        starts.append((f"RANDOM_{k+1}", repair(problem, a)))

    best_a = np.zeros(problem.n)
    best_u = problem.utility(best_a)
    diag = []

    for name, start in starts:
        a, u, passes = coordinate_ascent(problem, start, rng)
        diag.append({
            "start": name,
            "expected_log_wealth": u,
            "certainty_equivalent_wealth": float(np.exp(u)) if np.isfinite(u) else np.nan,
            "selected_trades": int(np.count_nonzero(a)),
            "defined_risk_dollars": float(problem.risk @ a),
            "passes": passes,
            "actions": ",".join(f"{x:.2f}" for x in a),
        })
        if u > best_u + 1e-12:
            best_a, best_u = a.copy(), u

    return best_a, best_u, pd.DataFrame(diag).sort_values("expected_log_wealth", ascending=False)


def metrics(name, a, problem):
    w = problem.wealth(a)
    u = problem.utility(a)
    return {
        "method": name,
        "expected_log_wealth": u,
        "certainty_equivalent_wealth": float(np.exp(u)),
        "mean_terminal_wealth": float(w.mean()),
        "median_terminal_wealth": float(np.median(w)),
        "p05_terminal_wealth": float(np.quantile(w, .05)),
        "minimum_terminal_wealth": float(w.min()),
        "probability_of_loss": float(np.mean(w < problem.equity)),
        "expected_pnl_after_cost": float(w.mean()-problem.equity),
        "transaction_cost_handling": "ALREADY_EMBEDDED_IN_MC_SCENARIO_PNL",
        "defined_risk_dollars": float(problem.risk @ a),
        "selected_trades": int(np.count_nonzero(a)),
    }


def size_trades(algorithm, trades: pd.DataFrame, scenarios: pd.DataFrame):
    if trades is None or trades.empty:return pd.DataFrame(),pd.DataFrame()
    trades=trades.copy().reset_index(drop=True)
    equity,current_risk_pct,ticker_risk_pct=portfolio_state(algorithm)
    problem=load_problem(trades,scenarios,equity,current_risk_pct,ticker_risk_pct)
    if problem.n==0:
        out=trades.copy();out["hjb_action_size"]=0.0;out["hjb_position_pct"]=0.0;out["hjb_decision"]="NO_TRADE";return out,pd.DataFrame()
    a,u,diag=optimize(problem,trades);by_index=dict(zip(problem.indices,map(float,a)));rows=[]
    for i,r in trades.iterrows():
        x=by_index.get(i,0.0);d=r.to_dict()
        default_per_action=UNCAPPED_POSITION_PCT_PER_ACTION if str(r.get("risk_class","CAPPED")).upper()=="UNCAPPED" else POSITION_PCT_PER_ACTION
        per_action=row_float(r,"hjb_position_pct_per_action",default_per_action)
        d.update({
            "discrete_log_wealth_action_size":x,"discrete_log_wealth_position_pct":x*per_action,"discrete_log_wealth_decision":decision(x),
            "hjb_action_size":x,"hjb_position_pct":x*per_action,"hjb_optimizer_value":u,"hjb_decision":decision(x),"hjb_trade_valid":i in problem.indices,
            "hjb_estimated_trade_risk_pct":100.0*x*risk_full(r)/equity,"hjb_method":"DISCRETE_EXPECTED_LOG_WEALTH","hjb_available_risk_pct":max(MAX_TOTAL_DEFINED_RISK_PCT-current_risk_pct,0.0),
        });rows.append(d)
    result=pd.DataFrame(rows).sort_values(["hjb_action_size","mc_expected_pnl"],ascending=[False,False]).reset_index(drop=True)
    return result,diag


