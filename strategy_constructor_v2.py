import json
import math
import numpy as np
import pandas as pd
import qc_config_v2 as cfg


def _row(chain,ticker,expiry,opt_type,strike):
    x=chain[(chain['ticker']==ticker)&(chain['expiry']==expiry)&(chain['option_type']==opt_type)&(chain['strike'].astype(float)==float(strike))]
    return None if x.empty else x.iloc[-1]


def _neighbors(chain,ticker,expiry,opt_type,target):
    x=chain[(chain['ticker']==ticker)&(chain['expiry']==expiry)&(chain['option_type']==opt_type)].copy()
    ks=sorted(x['strike'].dropna().astype(float).unique())
    if len(ks)<3: return None
    k=min(ks,key=lambda z:abs(z-float(target))); i=ks.index(k)
    # Mechanical robustness: if the strongest signal is on the quoted wing,
    # shift the fly center to the nearest interior strike instead of discarding it.
    i=max(1,min(i,len(ks)-2)); k=ks[i]
    return ks[i-1],k,ks[i+1]


def _leg(r,qty):
    return {'side':'BUY' if qty>0 else 'SELL','option_type':str(r['option_type']),'strike':float(r['strike']),'qty':abs(int(qty)),'signed_qty':int(qty),'mid':float(r['mid']),'bid':float(r['bid']),'ask':float(r['ask']),'iv':float(r['implied_volatility'])}


def _expiry_pnl(legs, signed_package, grid):
    out=np.zeros(len(grid),float)
    for l in legs:
        K=float(l['strike']); q=int(l['signed_qty'])
        intrinsic=np.maximum(grid-K,0.0) if str(l['option_type']).startswith('c') else np.maximum(K-grid,0.0)
        out += q*intrinsic
    return out - float(signed_package)


def construct_strategies(surface, signals, target_dte=20):
    if surface is None or surface.empty or signals is None or signals.empty:
        return pd.DataFrame()
    rows=[]
    for _,s in signals.iterrows():
        ticker=str(s['ticker']); expiry=pd.Timestamp(s['expiry']); opt=str(s['option_type']); target=float(s['strike'])
        n=_neighbors(surface,ticker,expiry,opt,target)
        if n is None: continue
        k1,k2,k3=n
        rs=[_row(surface,ticker,expiry,opt,k) for k in [k1,k2,k3]]
        if any(r is None for r in rs): continue
        if str(s['signal_side'])=='SHORT_RICH':
            qty=[1,-2,1]; structure='SELL_RICH_LOCAL_FLY'; vol_side='SHORT_VOL'
        else:
            qty=[-1,2,-1]; structure='BUY_CHEAP_LOCAL_FLY'; vol_side='LONG_VOL'
        legs=[_leg(r,q) for r,q in zip(rs,qty)]
        signed_package=float(sum(l['signed_qty']*l['mid'] for l in legs))
        ptype='DEBIT' if signed_package>0 else 'CREDIT'
        net_debit=max(signed_package,0.0); net_credit=max(-signed_package,0.0)
        spot=float(s['underlying_price']); grid=np.linspace(max(.01,.20*spot),2.0*spot,2001)
        pnl=_expiry_pnl(legs,signed_package,grid); max_profit=float(np.max(pnl)); max_loss=float(abs(np.min(pnl)))
        if not np.isfinite(max_loss) or max_loss<=.01 or not np.isfinite(max_profit) or max_profit<=.01: continue
        symbols=[r['symbol'] for r in rs]
        row={
            'snapshot_date':pd.Timestamp(s['snapshot_date']),'date':pd.Timestamp(s['snapshot_date']),'ticker':ticker,'structure':structure,'expiry':expiry,'underlying_price':spot,
            'vol_side':vol_side,'premium_type':ptype,'net_credit':net_credit,'net_debit':net_debit,'max_loss':max_loss,'max_profit':max_profit,
            'legs_json':json.dumps(legs),'lean_leg_symbols':symbols,'lean_signed_qty':qty,
            'signal_side':str(s['signal_side']),'target_option_type':opt,'target_strike':k2,'market_iv':float(s['market_iv']),'fair_iv':float(s['fair_iv']),
            'surface_residual':float(s['surface_residual']),'surface_z':float(s['surface_z']),'forecast_vol_20d':float(s['forecast_vol_20d']),
            'forecast_variance_20d':float(s['forecast_variance_20d']),'iv_rv_ratio':float(s['iv_rv_ratio']),'variance_premium':float(s['variance_premium']),
            'sabr_alpha':float(s['sabr_alpha']),'sabr_beta':float(s['sabr_beta']),'sabr_rho':float(s['sabr_rho']),'sabr_nu':float(s['sabr_nu']),'sabr_rmse':float(s['sabr_rmse']),
            'event_variance_20d':float(s.get('event_variance_20d',0.0) or 0.0),'jump_variance_20d':float(s.get('jump_variance_20d',0.0) or 0.0),
            'relative_vol_bucket':str(s['signal_side']),'variance_premium_ratio':float(s.get('variance_premium_ratio',float(s['iv_rv_ratio'])**2)),'cross_sectional_vrp_score':float(s['surface_z']),
        }
        rows.append(row)
    return pd.DataFrame(rows)



def filter_liquid_strategies(strategies, max_relative_spread=None):
    """Remove structures whose displayed edge is not executable.

    Uses the same package sign convention as execution_v2:
    positive signed package = debit, negative = credit.
    """
    if strategies is None or strategies.empty:
        return pd.DataFrame() if strategies is None else strategies.copy()

    kept=[]
    for _,row in strategies.iterrows():
        try:
            legs=json.loads(row['legs_json'])
            if not legs:
                continue
            signed_mid=0.0; signed_natural=0.0; max_rel=0.0
            ok=True
            for leg in legs:
                q=int(leg['signed_qty']); bid=float(leg['bid']); ask=float(leg['ask']); mid=float(leg['mid'])
                if not np.isfinite(bid) or not np.isfinite(ask) or not np.isfinite(mid): ok=False; break
                if bid<=0.0 or ask<=0.0 or ask<bid or mid<=0.0: ok=False; break
                rel=(ask-bid)/mid
                if max_relative_spread is not None and rel>float(max_relative_spread):
                    if (ask-bid)>float(getattr(cfg,'LIQUIDITY_ABSOLUTE_SPREAD_OVERRIDE',0.03)): ok=False; break
                max_rel=max(max_rel,float(rel))
                signed_mid += q*mid
                signed_natural += q*(ask if q>0 else bid)
            if not ok:
                continue

            ptype=str(row['premium_type']).upper()
            if ptype=='CREDIT':
                mid_credit=-signed_mid; natural_credit=-signed_natural
                if mid_credit<=0.0 or natural_credit<=0.0:
                    continue
                floor=max(mid_credit*float(cfg.MIN_CREDIT_FRACTION_OF_MID),float(cfg.MIN_NET_CREDIT))
                if natural_credit<floor:
                    continue
            else:
                if signed_mid<=0.0 or signed_natural<=0.0:
                    continue
                if signed_natural > signed_mid*float(getattr(cfg,'MAX_DEBIT_MULTIPLE_OF_MID',1.25)):
                    continue

            # Research/HJB economics use the CURRENT executable natural package.
            # The separate order-limit cushion is only a marketability guardrail;
            # it is not an assumed fill price and must not erase valid edge.
            mode=str(getattr(cfg,'ENTRY_EXECUTION_MODE','NATURAL_MARKETABLE')).upper()
            exec_signed=float(signed_natural) if mode=='NATURAL_MARKETABLE' else float(signed_mid)
            if ptype=='CREDIT':
                if -exec_signed < floor:
                    continue
            else:
                if exec_signed <= 0.0:
                    continue

            spot=float(row['underlying_price'])
            risk_class=str(row.get('risk_class','CAPPED')).upper()
            if risk_class=='UNCAPPED':
                exec_max_profit=max(-float(exec_signed),0.0)
                exec_max_loss=float(row.get('risk_proxy_per_share',row.get('max_loss',0.0)) or 0.0)
            else:
                grid=np.linspace(max(.01,.20*spot),2.0*spot,2001)
                pnl_exec=_expiry_pnl(legs,exec_signed,grid)
                exec_max_profit=float(np.max(pnl_exec)); exec_max_loss=float(abs(np.min(pnl_exec)))
            if not np.isfinite(exec_max_loss) or exec_max_loss<=.01 or not np.isfinite(exec_max_profit) or exec_max_profit<=.01:
                continue

            d=row.to_dict()
            d.setdefault('risk_class','CAPPED')
            d.setdefault('is_capped',True)
            d['model_mid_max_loss']=float(row.get('max_loss',np.nan)); d['model_mid_max_profit']=float(row.get('max_profit',np.nan))
            d['max_loss']=exec_max_loss; d['max_profit']=exec_max_profit
            d['entry_mid_signed']=float(signed_mid); d['entry_natural_signed']=float(signed_natural)
            d['entry_execution_signed']=exec_signed; d['max_leg_relative_spread']=float(max_rel)
            d['liquidity_spread_cap']=float(max_relative_spread) if max_relative_spread is not None else np.nan
            kept.append(d)
        except Exception:
            continue
    return pd.DataFrame(kept)



def _norm_cdf(x):
    return 0.5*(1.0+math.erf(float(x)/math.sqrt(2.0)))


def _delta_for_contract(row, spot, dte):
    K=max(float(row['strike']),1e-9)
    sigma=max(float(row['implied_volatility']),1e-6)
    T=max(float(dte)/365.0,1.0/365.0)
    r=float(getattr(cfg,'RISK_FREE_RATE',0.04))
    d1=(math.log(max(float(spot),1e-9)/K)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    if str(row['option_type']).lower().startswith('c'):
        return _norm_cdf(d1)
    return _norm_cdf(d1)-1.0


def _pick_strangle_leg(surface, ticker, expiry, option_type, spot, target_delta):
    x=surface[
        (surface['ticker'].astype(str)==str(ticker))&
        (pd.to_datetime(surface['expiry']).dt.normalize()==pd.Timestamp(expiry).normalize())&
        (surface['option_type'].astype(str)==str(option_type))
    ].copy()
    if x.empty:
        return None
    if str(option_type).startswith('c'):
        x=x[x['strike'].astype(float)>float(spot)].copy()
        signed_target=abs(float(target_delta))
    else:
        x=x[x['strike'].astype(float)<float(spot)].copy()
        signed_target=-abs(float(target_delta))
    if x.empty:
        return None
    dte=max(int((pd.Timestamp(expiry).normalize()-pd.Timestamp(x['snapshot_date'].iloc[0]).normalize()).days),1)
    x['_delta']=[_delta_for_contract(r,float(spot),dte) for _,r in x.iterrows()]
    x['_distance']=(x['_delta'].astype(float)-signed_target).abs()
    lo=float(getattr(cfg,'STRANGLE_MIN_ABS_DELTA',0.08))
    hi=float(getattr(cfg,'STRANGLE_MAX_ABS_DELTA',0.25))
    band=x[x['_delta'].abs().between(lo,hi)].copy()
    if not band.empty:
        x=band
    return x.sort_values(['_distance','relative_bid_ask_spread' if 'relative_bid_ask_spread' in x.columns else 'strike']).iloc[0]


def construct_short_strangles(surface, signals, target_dte=20):
    """Construct one-delta-pair short strangles from broader absolute-vol carry."""
    if surface is None or surface.empty or signals is None or signals.empty:
        return pd.DataFrame()
    rows=[]
    target_delta=float(getattr(cfg,'STRANGLE_TARGET_DELTA',0.16))
    for _,s in signals.iterrows():
        ticker=str(s['ticker']); expiry=pd.Timestamp(s['expiry']).normalize()
        spot=float(s['underlying_price'])
        call=_pick_strangle_leg(surface,ticker,expiry,'call',spot,target_delta)
        put=_pick_strangle_leg(surface,ticker,expiry,'put',spot,target_delta)
        if call is None or put is None:
            continue
        ref_vol=max(float(s.get('strangle_reference_vol',s.get('forecast_vol_20d',0.0)) or 0.0),1e-9)
        pair_iv=0.5*(float(call['implied_volatility'])+float(put['implied_volatility']))
        pair_ratio=pair_iv/ref_vol
        if pair_ratio<float(getattr(cfg,'STRANGLE_MIN_IV_RV',1.0)):
            continue
        legs=[_leg(put,-1),_leg(call,-1)]
        signed_package=float(sum(l['signed_qty']*l['mid'] for l in legs))
        if signed_package>=0:
            continue
        net_credit=-signed_package
        risk_proxy=max(
            float(getattr(cfg,'UNCAPPED_MARGIN_NOTIONAL_PCT',0.12))*spot,
            2.0*net_credit,
        )
        symbols=[put['symbol'],call['symbol']]
        row={
            'snapshot_date':pd.Timestamp(s['snapshot_date']),'date':pd.Timestamp(s['snapshot_date']),
            'ticker':ticker,'structure':'SHORT_STRANGLE','expiry':expiry,'underlying_price':spot,
            'vol_side':'SHORT_VOL','premium_type':'CREDIT','net_credit':net_credit,'net_debit':0.0,
            # max_loss is deliberately a sizing/margin proxy for the uncapped trade.
            # true tail risk remains in MC and is explicitly tagged as uncapped.
            'max_loss':float(risk_proxy),'max_profit':float(net_credit),
            'risk_proxy_per_share':float(risk_proxy),'risk_class':'UNCAPPED','is_capped':False,
            'true_max_loss':np.inf,
            'legs_json':json.dumps(legs),'lean_leg_symbols':symbols,'lean_signed_qty':[-1,-1],
            'signal_side':'SHORT_STRANGLE','target_option_type':'strangle',
            'target_strike':float(spot),'market_iv':float(pair_iv),'fair_iv':float(s['fair_iv']),
            'surface_residual':float(s.get('surface_residual',0.0) or 0.0),
            'surface_z':float(s.get('surface_z',0.0) or 0.0),
            'forecast_vol_20d':float(s['forecast_vol_20d']),
            'forecast_variance_20d':float(s['forecast_variance_20d']),
            'continuous_variance_20d':float(s.get('continuous_variance_20d',s['forecast_variance_20d'])),
            'iv_rv_ratio':float(pair_ratio),'variance_premium':float(pair_iv*pair_iv-ref_vol*ref_vol),
            'sabr_alpha':float(s['sabr_alpha']),'sabr_beta':float(s['sabr_beta']),
            'sabr_rho':float(s['sabr_rho']),'sabr_nu':float(s['sabr_nu']),'sabr_rmse':float(s['sabr_rmse']),
            'event_variance_20d':float(s.get('event_variance_20d',0.0) or 0.0),
            'jump_variance_20d':float(s.get('jump_variance_20d',0.0) or 0.0),
            'relative_vol_bucket':'ABSOLUTE_STRANGLE_VRP',
            'variance_premium_ratio':float(pair_ratio*pair_ratio),
            'cross_sectional_vrp_score':float(s.get('cross_sectional_vrp_score',s.get('iv_rv_ratio',1.0)-1.0)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# ASYM_V5: independent OTM put / call carry candidates
# ============================================================================

def _side_budget_fields(role):
    role=str(role).upper()
    if role=='CALL':
        return {
            'hjb_position_pct_per_action':float(cfg.ASYM_CALL_POSITION_PCT_PER_ACTION),
            'hjb_max_single_risk_pct':float(cfg.ASYM_CALL_MAX_SINGLE_TRADE_RISK_PCT),
            'hjb_max_ticker_risk_pct':float(cfg.ASYM_CALL_MAX_TICKER_RISK_PCT),
            'max_contracts_per_trade':int(cfg.ASYM_CALL_MAX_CONTRACTS_PER_TRADE),
        }
    return {
        'hjb_position_pct_per_action':float(cfg.ASYM_PUT_POSITION_PCT_PER_ACTION),
        'hjb_max_single_risk_pct':float(cfg.ASYM_PUT_MAX_SINGLE_TRADE_RISK_PCT),
        'hjb_max_ticker_risk_pct':float(cfg.ASYM_PUT_MAX_TICKER_RISK_PCT),
        'max_contracts_per_trade':int(cfg.ASYM_PUT_MAX_CONTRACTS_PER_TRADE),
    }


def construct_short_vol_legs(surface, signals, target_dte=20):
    """Build OTM short PUT and CALL candidates as separate research rows.

    This replaces the old assumption that a broad VRP signal must be expressed as
    a two-leg strangle.  The put and call are selected from the SAME delta pair,
    but each receives its own IV/RV economics and later its own MC/HJB decision.

    A standalone put deliberately retains the *full pair-style* risk proxy.  This
    is the live-project implementation of the C3 ablation result: removing the call
    must not also halve the put's management risk basis and thereby tighten exits.
    """
    if surface is None or surface.empty or signals is None or signals.empty:
        return pd.DataFrame()

    rows=[]
    target_delta=float(getattr(cfg,'STRANGLE_TARGET_DELTA',0.16))
    min_ratio=float(getattr(cfg,'STRANGLE_MIN_IV_RV',1.0))
    package_only=bool(getattr(cfg,'ASYM_PACKAGE_ONLY_MANAGEMENT',True))

    for _,s in signals.iterrows():
        ticker=str(s['ticker'])
        expiry=pd.Timestamp(s['expiry']).normalize()
        spot=float(s['underlying_price'])
        call=_pick_strangle_leg(surface,ticker,expiry,'call',spot,target_delta)
        put=_pick_strangle_leg(surface,ticker,expiry,'put',spot,target_delta)
        if call is None or put is None:
            continue

        ref_vol=max(float(s.get('strangle_reference_vol',s.get('forecast_vol_20d',0.0)) or 0.0),1e-9)
        pair_mid_credit=max(float(put['mid']),0.0)+max(float(call['mid']),0.0)
        full_risk_proxy=max(
            float(getattr(cfg,'UNCAPPED_MARGIN_NOTIONAL_PCT',0.12))*spot,
            2.0*pair_mid_credit,
        )
        pair_key=f"{ticker}|{expiry.date()}"

        for role,legrow in (('PUT',put),('CALL',call)):
            leg_iv=float(legrow['implied_volatility'])
            ratio=leg_iv/ref_vol
            variance_premium=leg_iv*leg_iv-ref_vol*ref_vol
            # Independent pre-MC gate: a side must itself be at least rich versus
            # the same continuous-RV reference.  The opposite leg cannot rescue it.
            if ratio<min_ratio or variance_premium<=0.0:
                continue

            leg=_leg(legrow,-1)
            signed_mid=-float(leg['mid'])
            if signed_mid>=0.0:
                continue
            net_credit=-signed_mid
            budget=_side_budget_fields(role)
            row={
                'snapshot_date':pd.Timestamp(s['snapshot_date']),'date':pd.Timestamp(s['snapshot_date']),
                'ticker':ticker,'structure':f'SHORT_{role}_VRP','expiry':expiry,'underlying_price':spot,
                'vol_side':'SHORT_VOL','premium_type':'CREDIT','net_credit':net_credit,'net_debit':0.0,
                # FULL pair-style risk basis for BOTH research sides.  Position size
                # is differentiated by the row-level HJB budget, not by shrinking
                # the put's stop/trailing denominator.
                'max_loss':float(full_risk_proxy),'max_profit':float(net_credit),
                'risk_proxy_per_share':float(full_risk_proxy),'management_risk_basis_per_share':float(full_risk_proxy),
                'risk_class':'UNCAPPED','is_capped':False,'true_max_loss':np.inf,
                'package_only_management':package_only,
                'option_leg_role':role,'independent_leg_candidate':True,'pair_key':pair_key,
                'legs_json':json.dumps([leg]),'lean_leg_symbols':[legrow['symbol']],'lean_signed_qty':[-1],
                'signal_side':f'SHORT_{role}_VRP','target_option_type':str(legrow['option_type']),
                'target_strike':float(legrow['strike']),'market_iv':leg_iv,'fair_iv':float(s['fair_iv']),
                'surface_residual':float(s.get('surface_residual',0.0) or 0.0),
                'surface_z':float(s.get('surface_z',0.0) or 0.0),
                'forecast_vol_20d':float(s['forecast_vol_20d']),
                'forecast_variance_20d':float(s['forecast_variance_20d']),
                'continuous_variance_20d':float(s.get('continuous_variance_20d',s['forecast_variance_20d'])),
                'iv_rv_ratio':float(ratio),'variance_premium':float(variance_premium),
                'sabr_alpha':float(s['sabr_alpha']),'sabr_beta':float(s['sabr_beta']),
                'sabr_rho':float(s['sabr_rho']),'sabr_nu':float(s['sabr_nu']),'sabr_rmse':float(s['sabr_rmse']),
                'event_variance_20d':float(s.get('event_variance_20d',0.0) or 0.0),
                'jump_variance_20d':float(s.get('jump_variance_20d',0.0) or 0.0),
                'relative_vol_bucket':f'ABSOLUTE_{role}_VRP',
                'variance_premium_ratio':float(ratio*ratio),
                'cross_sectional_vrp_score':float(ratio-1.0),
                'pair_mid_credit':float(pair_mid_credit),
            }
            row.update(budget)
            rows.append(row)

    return pd.DataFrame(rows)


def _one_selected(df, ticker, expiry):
    if df is None or df.empty:
        return None
    x=df[
        (df['ticker'].astype(str)==str(ticker)) &
        (pd.to_datetime(df['expiry']).dt.normalize()==pd.Timestamp(expiry).normalize())
    ].copy()
    if x.empty:
        return None
    if 'mc_expected_pnl' in x.columns:
        x=x.sort_values('mc_expected_pnl',ascending=False)
    return x.iloc[0]


def combine_independent_leg_selections(put_selected, call_selected):
    """Translate independent HJB approvals into executable rows.

    - PUT only approved  -> standalone SHORT_PUT_VRP.
    - CALL only approved -> standalone SHORT_CALL_VRP with the smaller call budget.
    - BOTH approved      -> one SHORT_STRANGLE_APPROVED package, call-budget limited.
    - Neither approved   -> no row / cash.

    No side is admitted because the opposite side passed.  The pairing step occurs
    only AFTER separate MC/HJB decisions have already been made.
    """
    p=put_selected.copy() if put_selected is not None else pd.DataFrame()
    c=call_selected.copy() if call_selected is not None else pd.DataFrame()
    keys=set()
    for df in (p,c):
        if df is not None and not df.empty:
            for _,r in df.iterrows():
                keys.add((str(r['ticker']),pd.Timestamp(r['expiry']).normalize()))

    out=[]
    for ticker,expiry in sorted(keys,key=lambda z:(z[0],z[1])):
        pr=_one_selected(p,ticker,expiry)
        cr=_one_selected(c,ticker,expiry)
        if pr is None and cr is None:
            continue
        if pr is None:
            # V7: do not express a naked short call.  The joint HJB result is
            # retained in diagnostics, but execution requires an independently
            # approved PUT on the same ticker/expiry so that the call is paired.
            if not bool(getattr(cfg, 'ASYM_ALLOW_STANDALONE_CALL', False)):
                continue
            d=cr.to_dict(); d['paired_independent_approval']=False
            d['independent_put_hjb_action']=0.0; d['independent_call_hjb_action']=float(d.get('hjb_action_size',0.0))
            out.append(d); continue
        if cr is None:
            d=pr.to_dict(); d['paired_independent_approval']=False
            d['independent_put_hjb_action']=float(d.get('hjb_action_size',0.0)); d['independent_call_hjb_action']=0.0
            out.append(d); continue

        pdict=pr.to_dict(); cdict=cr.to_dict(); d=dict(pdict)
        plegs=json.loads(pdict['legs_json']) if isinstance(pdict['legs_json'],str) else list(pdict['legs_json'])
        clegs=json.loads(cdict['legs_json']) if isinstance(cdict['legs_json'],str) else list(cdict['legs_json'])
        legs=plegs+clegs
        d.update({
            'structure':'SHORT_STRANGLE_APPROVED','signal_side':'SHORT_STRANGLE_APPROVED',
            'option_leg_role':'STRANGLE','target_option_type':'strangle','target_strike':float(pdict['underlying_price']),
            'legs_json':json.dumps(legs),
            'lean_leg_symbols':list(pdict['lean_leg_symbols'])+list(cdict['lean_leg_symbols']),
            'lean_signed_qty':list(pdict['lean_signed_qty'])+list(cdict['lean_signed_qty']),
            'net_credit':float(pdict.get('net_credit',0.0))+float(cdict.get('net_credit',0.0)),
            'net_debit':0.0,
            'max_loss':max(float(pdict.get('max_loss',0.0)),float(cdict.get('max_loss',0.0))),
            'risk_proxy_per_share':max(float(pdict.get('risk_proxy_per_share',0.0)),float(cdict.get('risk_proxy_per_share',0.0))),
            'management_risk_basis_per_share':max(float(pdict.get('management_risk_basis_per_share',0.0)),float(cdict.get('management_risk_basis_per_share',0.0))),
            'max_profit':float(pdict.get('max_profit',0.0))+float(cdict.get('max_profit',0.0)),
            'entry_mid_signed':float(pdict.get('entry_mid_signed',-float(pdict.get('net_credit',0.0))))+float(cdict.get('entry_mid_signed',-float(cdict.get('net_credit',0.0)))),
            'entry_natural_signed':float(pdict.get('entry_natural_signed',-float(pdict.get('net_credit',0.0))))+float(cdict.get('entry_natural_signed',-float(cdict.get('net_credit',0.0)))),
            'entry_execution_signed':float(pdict.get('entry_execution_signed',-float(pdict.get('net_credit',0.0))))+float(cdict.get('entry_execution_signed',-float(cdict.get('net_credit',0.0)))),
            'market_iv':0.5*(float(pdict.get('market_iv',0.0))+float(cdict.get('market_iv',0.0))),
            'iv_rv_ratio':0.5*(float(pdict.get('iv_rv_ratio',0.0))+float(cdict.get('iv_rv_ratio',0.0))),
            'variance_premium':float(pdict.get('variance_premium',0.0))+float(cdict.get('variance_premium',0.0)),
            # Diagnostics only: independent HJB already made the eligibility decision.
            'mc_expected_pnl':float(pdict.get('mc_expected_pnl',0.0))+float(cdict.get('mc_expected_pnl',0.0)),
            'mc_p25_scenario_expected_pnl':float(pdict.get('mc_p25_scenario_expected_pnl',0.0))+float(cdict.get('mc_p25_scenario_expected_pnl',0.0)),
            'mc_worst_case_expected_pnl':float(pdict.get('mc_worst_case_expected_pnl',0.0))+float(cdict.get('mc_worst_case_expected_pnl',0.0)),
            'mc_cvar_95':float(pdict.get('mc_cvar_95',0.0))+float(cdict.get('mc_cvar_95',0.0)),
            'mc_pop':min(float(pdict.get('mc_pop',0.0)),float(cdict.get('mc_pop',0.0))),
            'hjb_action_size':min(float(pdict.get('hjb_action_size',0.0)),float(cdict.get('hjb_action_size',0.0))),
            'hjb_position_pct':min(float(pdict.get('hjb_position_pct',0.0)),float(cdict.get('hjb_position_pct',0.0))),
            'hjb_decision':'TRADE_STRANGLE_BOTH_APPROVED',
            'hjb_position_pct_per_action':float(cfg.ASYM_STRANGLE_POSITION_PCT_PER_ACTION),
            'hjb_max_single_risk_pct':float(cfg.ASYM_STRANGLE_MAX_SINGLE_TRADE_RISK_PCT),
            'hjb_max_ticker_risk_pct':float(cfg.ASYM_STRANGLE_MAX_TICKER_RISK_PCT),
            'max_contracts_per_trade':int(cfg.ASYM_STRANGLE_MAX_CONTRACTS_PER_TRADE),
            'package_only_management':bool(getattr(cfg,'ASYM_PACKAGE_ONLY_MANAGEMENT',True)),
            'paired_independent_approval':True,
            'independent_put_hjb_action':float(pdict.get('hjb_action_size',0.0)),
            'independent_call_hjb_action':float(cdict.get('hjb_action_size',0.0)),
            'put_mc_expected_pnl':float(pdict.get('mc_expected_pnl',0.0)),
            'call_mc_expected_pnl':float(cdict.get('mc_expected_pnl',0.0)),
        })
        out.append(d)

    result=pd.DataFrame(out)
    if result.empty:
        return result
    # execution_v2 intentionally permits one open structure per ticker.  Make the
    # choice deterministic instead of relying on row order when several expiries pass.
    result=result.sort_values(['mc_expected_pnl','hjb_position_pct'],ascending=False)
    return result.drop_duplicates('ticker',keep='first').reset_index(drop=True)
