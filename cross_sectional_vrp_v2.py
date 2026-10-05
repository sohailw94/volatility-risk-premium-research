import numpy as np
import pandas as pd
import qc_config_v2 as cfg


def build_cross_sectional_vrp(scored_surface, forward):
    """Original primary thesis signal: local SABR residual + absolute VRP.

    This function is intentionally unchanged in logic.  It remains the signal
    generator for the normal 10-35 DTE single-name universe.
    """
    if scored_surface is None or scored_surface.empty or forward is None or forward.empty:
        return pd.DataFrame()
    x=scored_surface.merge(forward,on='ticker',how='inner',suffixes=('','_fwd'))
    x=x.dropna(subset=['implied_volatility','sabr_fair_iv','surface_residual','surface_z','forecast_vol_20d']).copy()
    x['market_iv']=x['implied_volatility'].astype(float)
    x['fair_iv']=x['sabr_fair_iv'].astype(float)
    x['iv_rv_ratio']=x['market_iv']/x['forecast_vol_20d'].astype(float).clip(lower=1e-9)
    x['variance_premium']=x['market_iv']**2-x['forecast_variance_20d'].astype(float)
    x['signal_side']='NONE'

    rich=(x['surface_residual']>=cfg.MIN_SURFACE_RESIDUAL)&(x['surface_z']>=cfg.MIN_SURFACE_Z)&(x['iv_rv_ratio']>=cfg.MIN_SHORT_IV_RV)&(x['variance_premium']>0)
    cheap=(x['surface_residual']<=-cfg.MIN_SURFACE_RESIDUAL)&(x['surface_z']<=-cfg.MIN_SURFACE_Z)&(x['iv_rv_ratio']<=cfg.MAX_LONG_IV_RV)&(x['variance_premium']<0)
    if cfg.ENABLE_SHORT_RICH: x.loc[rich,'signal_side']='SHORT_RICH'
    if cfg.ENABLE_LONG_CHEAP: x.loc[cheap,'signal_side']='LONG_CHEAP'
    x=x[x['signal_side']!='NONE'].copy()
    if x.empty: return x

    # Exact audit rule: at most one rich and one cheap contract per ticker/day.
    rows=[]
    for ticker,g in x.groupby('ticker'):
        s=g[g['signal_side']=='SHORT_RICH']
        if not s.empty:
            rows.append(s.sort_values(['surface_z','variance_premium'],ascending=False).iloc[0])
        l=g[g['signal_side']=='LONG_CHEAP']
        if not l.empty:
            rows.append(l.sort_values(['surface_z','variance_premium'],ascending=True).iloc[0])
    out=pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    if not out.empty:
        out['relative_vol_bucket']=out['signal_side']
        out['variance_premium_ratio']=(out['market_iv']**2/out['forecast_variance_20d'].astype(float).clip(lower=1e-12)).clip(.01,20)
        out['cross_sectional_vrp_score']=out['surface_z']
    return out


def build_absolute_index_vrp(scored_surface, forward):
    """Short-dated SPY/NDX fallback signal.

    The fallback answers a different question from the primary signal:
    is liquid short-dated index IV rich versus the model's forward realized
    variance forecast?  It therefore does NOT require a local SABR residual or
    residual z-score to exceed the primary thresholds.

    SABR is still calibrated upstream and its fields are retained because the
    existing strategy/MC stack uses the fitted surface.  Only the *entry gate*
    differs.  We keep at most one near-ATM call and one near-ATM put per ticker
    so MC/HJB can compare liquid defined-risk implementations without exploding
    the candidate count.
    """
    if scored_surface is None or scored_surface.empty or forward is None or forward.empty:
        return pd.DataFrame()

    x=scored_surface.merge(forward,on='ticker',how='inner',suffixes=('','_fwd'))
    needed=['implied_volatility','sabr_fair_iv','surface_residual','surface_z',
            'forecast_vol_20d','forecast_variance_20d','strike','underlying_price']
    x=x.dropna(subset=needed).copy()
    if x.empty:
        return x

    x['market_iv']=x['implied_volatility'].astype(float)
    x['fair_iv']=x['sabr_fair_iv'].astype(float)
    fvol=x['forecast_vol_20d'].astype(float).clip(lower=1e-9)
    fvar=x['forecast_variance_20d'].astype(float).clip(lower=1e-12)
    x['iv_rv_ratio']=x['market_iv']/fvol
    x['variance_premium']=x['market_iv']**2-fvar
    x['signal_side']='NONE'

    # Absolute index VRP gate.  No local-SABR-residual requirement here.
    rich=(x['iv_rv_ratio']>=float(cfg.MIN_SHORT_IV_RV))&(x['variance_premium']>0)
    if bool(cfg.ENABLE_SHORT_RICH):
        x.loc[rich,'signal_side']='SHORT_RICH'
    x=x[x['signal_side']=='SHORT_RICH'].copy()
    if x.empty:
        return x

    strike=x['strike'].astype(float).clip(lower=1e-9)
    spot=x['underlying_price'].astype(float).clip(lower=1e-9)
    x['_atm_distance']=np.abs(np.log(strike/spot))

    # One near-ATM candidate per option type/ticker.  The option surface was
    # already restricted to real two-sided quotes and the fallback spread cap.
    rows=[]
    for (_, _),g in x.groupby(['ticker','option_type'],sort=False):
        g=g.sort_values(['_atm_distance','iv_rv_ratio','variance_premium'],
                        ascending=[True,False,False])
        rows.append(g.iloc[0])

    out=pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out=out.drop(columns=['_atm_distance'],errors='ignore')
    out['relative_vol_bucket']='ABSOLUTE_INDEX_VRP'
    out['variance_premium_ratio']=(out['market_iv']**2/out['forecast_variance_20d'].astype(float).clip(lower=1e-12)).clip(.01,20)
    # Keep the existing schema.  HJB/MC do not receive a fabricated SABR edge;
    # the score below expresses only absolute IV/RV richness for diagnostics.
    out['cross_sectional_vrp_score']=out['iv_rv_ratio']-1.0
    return out



def build_strangle_vrp(scored_surface, forward):
    """Broader absolute-carry signal for short strangles.

    Unlike the local-fly signal, this does not require an extreme SABR residual.
    The gate compares market IV with the continuous realized-vol forecast.  Jump
    and event risk are still retained in the production Monte Carlo, so this gate
    broadens candidate generation without pretending jump risk does not exist.
    """
    if scored_surface is None or scored_surface.empty or forward is None or forward.empty:
        return pd.DataFrame()

    x=scored_surface.merge(forward,on='ticker',how='inner',suffixes=('','_fwd'))
    needed=['implied_volatility','sabr_fair_iv','forecast_vol_20d',
            'forecast_variance_20d','strike','underlying_price','expiry']
    x=x.dropna(subset=needed).copy()
    if x.empty:
        return x

    x['market_iv']=x['implied_volatility'].astype(float)
    x['fair_iv']=x['sabr_fair_iv'].astype(float)

    if bool(getattr(cfg,'STRANGLE_USE_CONTINUOUS_RV',True)) and 'continuous_variance_20d' in x.columns:
        base_var=pd.to_numeric(x['continuous_variance_20d'],errors='coerce').clip(lower=1e-12)
        base_vol=np.sqrt(base_var)
    else:
        base_var=pd.to_numeric(x['forecast_variance_20d'],errors='coerce').clip(lower=1e-12)
        base_vol=pd.to_numeric(x['forecast_vol_20d'],errors='coerce').clip(lower=1e-9)

    x['strangle_reference_vol']=base_vol
    x['iv_rv_ratio']=x['market_iv']/base_vol.clip(lower=1e-9)
    x['variance_premium']=x['market_iv']**2-base_var
    x['signal_side']='NONE'

    rich=(x['iv_rv_ratio']>=float(getattr(cfg,'STRANGLE_MIN_IV_RV',1.0)))&(x['variance_premium']>0)
    if bool(cfg.ENABLE_SHORT_RICH):
        x.loc[rich,'signal_side']='SHORT_STRANGLE'
    x=x[x['signal_side']=='SHORT_STRANGLE'].copy()
    if x.empty:
        return x

    # One representative row per ticker/expiry.  The constructor uses the full
    # surface to choose OTM call/put strikes by delta.
    strike=x['strike'].astype(float).clip(lower=1e-9)
    spot=x['underlying_price'].astype(float).clip(lower=1e-9)
    x['_atm_distance']=np.abs(np.log(strike/spot))
    rows=[]
    for (ticker,expiry),g in x.groupby(['ticker','expiry'],sort=False):
        g=g.sort_values(['_atm_distance','iv_rv_ratio','variance_premium'],
                        ascending=[True,False,False])
        rows.append(g.iloc[0])

    out=pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out=out.drop(columns=['_atm_distance'],errors='ignore')
    out['relative_vol_bucket']='ABSOLUTE_STRANGLE_VRP'
    out['variance_premium_ratio']=(out['iv_rv_ratio'].astype(float)**2).clip(.01,20)
    out['cross_sectional_vrp_score']=out['iv_rv_ratio']-1.0
    out=out.sort_values(['iv_rv_ratio','variance_premium'],ascending=False)
    return out.head(int(getattr(cfg,'STRANGLE_MAX_CANDIDATES',12))).reset_index(drop=True)
