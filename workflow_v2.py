import pandas as pd
import qc_config_v2 as cfg
from qc_data_adapter_v2 import history_to_price_frame, current_option_surface
from forward_vol_model_v2 import build_forward_vol_features
from sabr_v2 import calibrate_surface
from cross_sectional_vrp_v2 import build_cross_sectional_vrp, build_absolute_index_vrp, build_strangle_vrp
from strategy_constructor_v2 import (
    construct_strategies,
    construct_short_vol_legs,
    combine_independent_leg_selections,
    filter_liquid_strategies,
)
from hmm_regime_model_v2 import latest_regimes_from_prices
from execution_policy_v2 import submit_hjb_combo_orders
import monte_carlo_multileg_v2 as mcmod
import hjb_position_sizer_v2 as hjbmod


def _run_mc(strategies):
    if hasattr(mcmod,'run_monte_carlo'):
        return mcmod.run_monte_carlo(strategies,n_paths=cfg.N_PATHS)
    return mcmod.run_mc(strategies,n_paths=cfg.N_PATHS)


def _size(algorithm,mc,scenarios):
    if hasattr(hjbmod,'size_trades'):
        out=hjbmod.size_trades(algorithm,mc,scenarios)
        return out if isinstance(out,tuple) else (out,pd.DataFrame())
    if hasattr(hjbmod,'optimize_hjb_clean_book'):
        return hjbmod.optimize_hjb_clean_book(mc,scenarios,float(algorithm.portfolio.total_portfolio_value)),pd.DataFrame()
    raise AttributeError('hjb_position_sizer_v2 has no supported sizing entry point')


def _empty_stack(mode='PRIMARY',dte=None,surface=None):
    e=pd.DataFrame()
    return {'forward':e,'sabr':e,'signals':e,'strategies_raw':e,'strategies':e,
            'surface':e if surface is None else surface,'mode':mode,'fallback_dte':dte}


def _signal_stack(algorithm,prices,surface,target_dte,max_relative_spread,mode='PRIMARY'):
    if surface is None or surface.empty:
        return _empty_stack(mode,target_dte if mode!='PRIMARY' else None,surface)

    forward=build_forward_vol_features(prices,surface,algorithm.time)
    sabr,scored=calibrate_surface(surface,target_dte)

    if mode=='PRIMARY':
        # Original capped local-SABR thesis remains unchanged.
        signals=build_cross_sectional_vrp(scored,forward)
        raw=construct_strategies(surface,signals,target_dte)
    elif mode=='INDEX_FLY':
        # Liquid short-dated fallback still has access to the capped fly expression.
        signals=build_absolute_index_vrp(scored,forward)
        raw=construct_strategies(surface,signals,target_dte)
    elif mode=='ASYM_CARRY':
        # Broader absolute carry, but PUT and CALL are now distinct research rows.
        signals=build_strangle_vrp(scored,forward)
        raw=construct_short_vol_legs(surface,signals,target_dte)
    else:
        raise ValueError(f'Unknown signal-stack mode: {mode}')

    strategies=filter_liquid_strategies(raw,max_relative_spread)
    return {'forward':forward,'sabr':sabr,'signals':signals,'strategies_raw':raw,
            'strategies':strategies,'surface':surface,'mode':mode,
            'fallback_dte':target_dte if mode!='PRIMARY' else None}


def _with_regimes(strategies,regimes):
    if strategies is None or strategies.empty:
        return pd.DataFrame() if strategies is None else strategies.copy()
    s=strategies.copy()
    if regimes is not None and not regimes.empty:
        s=s.merge(regimes.drop(columns=['date'],errors='ignore'),on='ticker',how='left')
    return s


def _research_only(algorithm,strategies):
    result={'mc':pd.DataFrame(),'hjb':pd.DataFrame(),'diag':pd.DataFrame(),
            'selected':pd.DataFrame(),'status':'NO_EXECUTABLE_STRATEGY'}
    if strategies is None or strategies.empty:
        return result
    mc,scenarios=_run_mc(strategies)
    result['mc']=mc
    if mc is None or mc.empty:
        result['status']='NO_MC_ROWS'
        return result
    hjb,diag=_size(algorithm,mc,scenarios)
    result['hjb']=hjb; result['diag']=diag
    selected=hjb[hjb['hjb_action_size']>0].copy() if hjb is not None and not hjb.empty and 'hjb_action_size' in hjb.columns else pd.DataFrame()
    result['selected']=selected
    result['status']='HJB_ZERO' if selected.empty else 'HJB_SELECTED'
    return result


def _prepare_stack(stack,regimes,min_exit_dte):
    strategies=_with_regimes(stack['strategies'],regimes)
    if not strategies.empty:
        strategies=strategies.copy()
        strategies['min_exit_dte']=int(min_exit_dte)
    stack['strategies']=strategies
    return strategies


def _evaluate(algorithm,stack,regimes,min_exit_dte):
    strategies=_prepare_stack(stack,regimes,min_exit_dte)
    research=_research_only(algorithm,strategies)
    result={'stack':stack,'mc':research['mc'],'hjb':research['hjb'],'diag':research['diag'],
            'selected':research['selected'],'tickets':[],'status':research['status']}
    if research['selected'].empty:
        return result
    for _,r in research['selected'].head(3).iterrows():
        algorithm.debug(
            f"V5_PICK|{stack.get('mode')}|T={r.get('ticker')}|S={r.get('structure')}|"
            f"EV={float(r.get('mc_expected_pnl',0.0)):.2f}|POP={float(r.get('mc_pop',0.0)):.2f}|"
            f"A={float(r.get('hjb_action_size',0.0)):.2f}|RISK={r.get('risk_class','CAPPED')}"
        )
    tickets=submit_hjb_combo_orders(algorithm,research['selected'],execute_orders=cfg.EXECUTE_ORDERS)
    result['tickets']=tickets
    result['status']='SUBMITTED' if tickets else 'SELECTED_NOT_SUBMITTED'
    return result


def _evaluate_asymmetric(algorithm,stack,regimes,min_exit_dte):
    """V6: one JOINT PUT/CALL/CASH optimization.

    All executable carry legs enter one MC scenario matrix and one HJB problem.
    Puts and calls therefore compete for the same portfolio risk budget and for
    expected-log-wealth improvement.  Pairing into a strangle occurs only AFTER
    this joint optimizer independently assigns a positive action to both legs.
    """
    strategies=_prepare_stack(stack,regimes,min_exit_dte)
    base={'stack':stack,'mc':pd.DataFrame(),'hjb':pd.DataFrame(),'diag':pd.DataFrame(),
          'selected':pd.DataFrame(),'tickets':[],'status':'NO_EXECUTABLE_STRATEGY',
          'put_selected':pd.DataFrame(),'call_selected':pd.DataFrame(),
          'put_mc':pd.DataFrame(),'call_mc':pd.DataFrame(),
          'put_hjb':pd.DataFrame(),'call_hjb':pd.DataFrame()}
    if strategies is None or strategies.empty:
        return base

    research=_research_only(algorithm,strategies)
    base['mc']=research['mc']; base['hjb']=research['hjb']; base['diag']=research['diag']

    def _split(df,role):
        if df is None or df.empty or 'option_leg_role' not in df.columns:
            return pd.DataFrame()
        return df[df['option_leg_role'].astype(str).str.upper()==role].copy()

    put_mc=_split(research['mc'],'PUT'); call_mc=_split(research['mc'],'CALL')
    put_hjb=_split(research['hjb'],'PUT'); call_hjb=_split(research['hjb'],'CALL')
    selected_all=research['selected'] if research['selected'] is not None else pd.DataFrame()
    put_sel=_split(selected_all,'PUT'); call_sel=_split(selected_all,'CALL')

    base['put_mc']=put_mc; base['call_mc']=call_mc
    base['put_hjb']=put_hjb; base['call_hjb']=call_hjb
    base['put_selected']=put_sel; base['call_selected']=call_sel

    executable=combine_independent_leg_selections(put_sel,call_sel)
    base['selected']=executable

    algorithm.debug(
        f"V7_JOINT|{algorithm.time.date()}|DTE={stack.get('fallback_dte')}|"
        f"rows={len(strategies)}|mc={len(research['mc'])}|hjb={len(research['hjb'])}|"
        f"put_sel={len(put_sel)}|call_sel={len(call_sel)}|exec={len(executable)}|"
        f"strangles={0 if executable.empty else int((executable['structure'].astype(str)=='SHORT_STRANGLE_APPROVED').sum())}"
    )

    if executable.empty:
        base['status']=research['status']
        return base

    for _,r in executable.head(5).iterrows():
        algorithm.debug(
            f"V7_JOINT_PICK|T={r.get('ticker')}|S={r.get('structure')}|ROLE={r.get('option_leg_role')}|"
            f"EV={float(r.get('mc_expected_pnl',0.0)):.2f}|A={float(r.get('hjb_action_size',0.0)):.2f}|"
            f"PUTA={float(r.get('independent_put_hjb_action',0.0)):.2f}|CALLA={float(r.get('independent_call_hjb_action',0.0)):.2f}|"
            f"POS={float(r.get('hjb_position_pct',0.0)):.2f}"
        )

    tickets=submit_hjb_combo_orders(algorithm,executable,execute_orders=cfg.EXECUTE_ORDERS)
    base['tickets']=tickets
    base['status']='SUBMITTED' if tickets else 'SELECTED_NOT_SUBMITTED'
    return base


def _attempt_summary(result,label):
    st=result['stack']; hjb=result['hjb']; sel=result['selected']
    return {
        'label':label,'mode':st.get('mode'),'dte':st.get('fallback_dte'),
        'surface_rows':len(st.get('surface',[])),'signals':len(st.get('signals',[])),
        'strategies_raw':len(st.get('strategies_raw',[])),'strategies':len(st.get('strategies',[])),
        'hjb_rows':0 if hjb is None else len(hjb),'selected':0 if sel is None else len(sel),
        'put_selected':len(result.get('put_selected',[])),'call_selected':len(result.get('call_selected',[])),
        'tickets':len(result.get('tickets',[])),'status':result.get('status','UNKNOWN')
    }


def _fallback_ticker_log(algorithm,result,dte,label):
    st=result['stack']; surface=st.get('surface',pd.DataFrame()); signals=st.get('signals',pd.DataFrame())
    strategies=st.get('strategies',pd.DataFrame()); selected=result.get('selected',pd.DataFrame())
    parts=[]
    for ticker in tuple(getattr(cfg,'FALLBACK_TICKERS',('SPY','NDX'))):
        def n(df):
            if df is None or df.empty or 'ticker' not in df.columns: return 0
            return int((df['ticker'].astype(str)==str(ticker)).sum())
        parts.append(f"{ticker}=srf{n(surface)}/sig{n(signals)}/liq{n(strategies)}/sel{n(selected)}")
    algorithm.debug(f"V5_FALLBACK|{algorithm.time.date()}|{label}|DTE={int(dte)}|"+'|'.join(parts)+f"|STATUS={result.get('status','UNKNOWN')}")


def _fallback_surfaces(algorithm,data):
    option_symbols=getattr(algorithm,'_vh_fallback_option_symbols',{}) or {}
    underlying_symbols=getattr(algorithm,'_vh_fallback_underlying_symbols',{}) or {}
    if not bool(getattr(cfg,'ENABLE_LIQUID_FALLBACK',False)) or not option_symbols or not underlying_symbols:
        return pd.DataFrame()
    return current_option_surface(
        algorithm,data,option_symbols,underlying_symbols,
        cfg.FALLBACK_MIN_DTE,cfg.FALLBACK_MAX_DTE,
        getattr(cfg,'FALLBACK_MAX_RELATIVE_BID_ASK_SPREAD',None),
    )


def _build_output(prices,result,attempts,primary):
    st=result['stack']
    return {
        'prices':prices,'surface':st.get('surface',pd.DataFrame()),'forward':st['forward'],
        'sabr':st['sabr'],'signals':st['signals'],'strategies':st['strategies'],
        'strategies_raw':st['strategies_raw'],'mc':result['mc'],'hjb':result['hjb'],
        'diag':result['diag'],'selected':result['selected'],'mode':st.get('mode','PRIMARY'),
        'fallback_dte':st.get('fallback_dte'),'decision_status':result['status'],
        'attempts':pd.DataFrame(attempts),'primary_strategies_raw':primary['strategies_raw'],
        'primary_strategies':primary['strategies'],
        'put_selected':result.get('put_selected',pd.DataFrame()),
        'call_selected':result.get('call_selected',pd.DataFrame()),
        'put_mc':result.get('put_mc',pd.DataFrame()),'call_mc':result.get('call_mc',pd.DataFrame()),
    }


def _decision_log(algorithm,attempts,final):
    bits=[]
    for a in attempts:
        side=f"/ps{a.get('put_selected',0)}/cs{a.get('call_selected',0)}" if a.get('mode')=='ASYM_CARRY' else ''
        bits.append(f"{a['label']}=srf{a['surface_rows']}/sig{a['signals']}/liq{a['strategies']}/sel{a['selected']}{side}/t{a['tickets']}/{a['status']}")
    algorithm.debug(f"V7_DECISION|{algorithm.time.date()}|"+'|'.join(bits)+f"|FINAL={final}")


def run_workflow(algorithm,data):
    prices=history_to_price_frame(algorithm,algorithm._vh_equity_symbols,cfg.PRICE_LOOKBACK_BARS)
    if prices.empty:
        algorithm.debug(f"V7_DECISION|{algorithm.time.date()}|FINAL=NO_PRICE_HISTORY")
        return {}
    regimes=latest_regimes_from_prices(prices,cfg.MIN_HISTORY_ROWS)

    primary_surface=current_option_surface(
        algorithm,data,algorithm._vh_option_symbols,algorithm._vh_equity_symbols,
        cfg.OPTION_MIN_DTE,cfg.OPTION_MAX_DTE,
        getattr(cfg,'PRIMARY_MAX_RELATIVE_BID_ASK_SPREAD',None),
    )
    primary=_signal_stack(algorithm,prices,primary_surface,cfg.TARGET_DTE,
                          getattr(cfg,'PRIMARY_MAX_RELATIVE_BID_ASK_SPREAD',None),'PRIMARY')
    attempts=[]

    # V6: evaluate the empirically stronger broad carry family FIRST.  PUT and
    # CALL rows compete in one joint HJB with CASH.  Only if carry submits no
    # executable order do we fall back to the legacy local-SABR fly branch.
    result={'tickets':[],'status':'NO_EXECUTABLE_STRATEGY','stack':primary,
            'mc':pd.DataFrame(),'hjb':pd.DataFrame(),'diag':pd.DataFrame(),'selected':pd.DataFrame()}
    if bool(getattr(cfg,'ENABLE_ASYMMETRIC_CARRY',False)):
        asym=_signal_stack(algorithm,prices,primary_surface,cfg.TARGET_DTE,
                           getattr(cfg,'PRIMARY_MAX_RELATIVE_BID_ASK_SPREAD',None),'ASYM_CARRY')
        result=_evaluate_asymmetric(algorithm,asym,regimes,cfg.MIN_EXIT_DTE)
        attempts.append(_attempt_summary(result,'A'))

    if not result['tickets'] and bool(getattr(cfg,'ENABLE_LEGACY_FLY_EXECUTION',False)):
        candidate=_evaluate(algorithm,primary,regimes,cfg.MIN_EXIT_DTE)
        attempts.append(_attempt_summary(candidate,'P'))
        result=candidate
    elif not result['tickets']:
        # V7 keeps the primary fly stack available in research output but does not
        # execute it.  This prevents a vetoed naked call from silently falling
        # through into a previously rejected trade family.
        attempts.append({'label':'P','mode':'PRIMARY','surface_rows':len(primary.get('surface',[])),
                         'signals':len(primary.get('signals',[])),'strategies':len(primary.get('strategies',[])),
                         'hjb_rows':0,'selected':0,'put_selected':0,'call_selected':0,
                         'tickets':0,'status':'RESEARCH_ONLY'})

    # Liquid short-dated fallback: carry first, capped index fly second.
    if not result['tickets'] and bool(getattr(cfg,'ENABLE_LIQUID_FALLBACK',False)):
        full=_fallback_surfaces(algorithm,data)
        for dte in tuple(getattr(cfg,'FALLBACK_DTES',(7,1))):
            surface=full[full['dte'].astype(int)==int(dte)].copy() if full is not None and not full.empty and 'dte' in full.columns else pd.DataFrame()

            if bool(getattr(cfg,'ENABLE_ASYMMETRIC_CARRY',False)):
                asym=_signal_stack(algorithm,prices,surface,int(dte),
                                   getattr(cfg,'FALLBACK_MAX_RELATIVE_BID_ASK_SPREAD',None),'ASYM_CARRY')
                candidate=_evaluate_asymmetric(algorithm,asym,regimes,getattr(cfg,'FALLBACK_EXIT_DTE',0))
                attempts.append(_attempt_summary(candidate,f'A{int(dte)}'))
                _fallback_ticker_log(algorithm,candidate,int(dte),'ASYM')
                result=candidate
                if candidate['tickets']:
                    break

            fly=_signal_stack(algorithm,prices,surface,int(dte),
                              getattr(cfg,'FALLBACK_MAX_RELATIVE_BID_ASK_SPREAD',None),'INDEX_FLY')
            if bool(getattr(cfg,'ENABLE_LEGACY_FLY_EXECUTION',False)):
                candidate=_evaluate(algorithm,fly,regimes,getattr(cfg,'FALLBACK_EXIT_DTE',0))
                attempts.append(_attempt_summary(candidate,f'F{int(dte)}'))
                _fallback_ticker_log(algorithm,candidate,int(dte),'FLY')
                result=candidate
                if candidate['tickets']:
                    break
            else:
                attempts.append({'label':f'F{int(dte)}','mode':'INDEX_FLY','surface_rows':len(fly.get('surface',[])),
                                 'signals':len(fly.get('signals',[])),'strategies':len(fly.get('strategies',[])),
                                 'hjb_rows':0,'selected':0,'put_selected':0,'call_selected':0,
                                 'tickets':0,'status':'RESEARCH_ONLY'})

    final='SUBMITTED' if result['tickets'] else result['status']
    _decision_log(algorithm,attempts,final)
    out=_build_output(prices,result,attempts,primary)
    algorithm._vh_last_research=out
    return out


def run_research_cycle(algorithm,config=None,data=None):
    data=data if data is not None else getattr(algorithm,'_vh_last_slice',None)
    return {} if data is None else run_workflow(algorithm,data)
