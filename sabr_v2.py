import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import qc_config_v2 as cfg


def sabr_iv(F, K, T, alpha, beta, rho, nu):
    F = max(float(F), 1e-9); K = max(float(K), 1e-9); T = max(float(T), 1e-9)
    if abs(F-K) < 1e-10:
        return float((alpha/F**(1-beta)) * (1 + (((1-beta)**2/24)*(alpha*alpha/F**(2-2*beta)) + (rho*beta*nu*alpha)/(4*F**(1-beta)) + ((2-3*rho*rho)/24)*nu*nu)*T))
    l = np.log(F/K); fb = (F*K)**((1-beta)/2); z = (nu/max(alpha,1e-12))*fb*l
    root = np.sqrt(max(1-2*rho*z+z*z,1e-12)); x = np.log((root+z-rho)/max(1-rho,1e-12))
    zx = 1.0 if abs(z) < 1e-10 else z/(x if abs(x)>1e-12 else 1e-12)
    den = fb*(1+((1-beta)**2/24)*l*l+((1-beta)**4/1920)*l**4)
    corr = 1+(((1-beta)**2/24)*(alpha*alpha/(F*K)**(1-beta))+(rho*beta*nu*alpha)/(4*fb)+((2-3*rho*rho)/24)*nu*nu)*T
    return float((alpha/den)*zx*corr)


def _robust_z(values):
    s = pd.Series(values, dtype=float)
    med = float(s.median()); mad = float((s-med).abs().median()); scale = 1.4826*mad
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(s.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.zeros(len(s))
    return ((s-med)/scale).to_numpy(float)


def _fit_one(g):
    g = g.copy(); g['implied_volatility'] = pd.to_numeric(g['implied_volatility'], errors='coerce')
    g = g[g['implied_volatility'].between(cfg.SABR_MIN_IV, cfg.SABR_MAX_IV)].copy()
    if len(g) < cfg.SABR_MIN_POINTS:
        return None
    F = float(g['underlying_price'].median()); T = max(float(g['dte'].median())/365.0, 1/365.0)
    K = g['strike'].to_numpy(float); market = g['implied_volatility'].to_numpy(float)
    atm = float(market[np.argmin(np.abs(K-F))]); beta = float(cfg.DEFAULT_SABR_BETA)
    x0 = np.array([np.clip(atm,.05,2.5),-.20,.60]); lo=np.array([.03,-.95,.02]); hi=np.array([3.0,.95,3.0])
    def resid(p):
        s,rho,nu=p; alpha=s*F**(1-beta)
        return np.array([sabr_iv(F,k,T,alpha,beta,rho,nu) for k in K])-market
    try:
        fit=least_squares(resid,x0,bounds=(lo,hi),loss='soft_l1',max_nfev=int(cfg.SABR_MAX_NFEV))
    except Exception:
        return None
    if not fit.success:
        return None
    s,rho,nu=fit.x; alpha=s*F**(1-beta)
    fitted=np.array([sabr_iv(F,k,T,alpha,beta,rho,nu) for k in K])
    rmse=float(np.sqrt(np.mean((fitted-market)**2)))
    if not np.isfinite(rmse) or rmse > cfg.SABR_MAX_ACCEPTED_RMSE:
        return None
    return {'alpha':float(alpha),'beta':beta,'rho':float(rho),'nu':float(nu),'rmse':rmse,'F':F,'T':T}


def calibrate_surface(surface, target_dte=None):
    target_dte = cfg.TARGET_DTE if target_dte is None else int(target_dte)
    if surface is None or surface.empty:
        return pd.DataFrame(), pd.DataFrame()
    cal_rows=[]; scored=[]
    for ticker,g0 in surface.groupby('ticker'):
        ex=g0[['expiry','dte']].drop_duplicates().copy()
        if ex.empty: continue
        i=(ex['dte'].astype(float)-target_dte).abs().idxmin(); expiry=pd.Timestamp(ex.loc[i,'expiry'])
        g=g0[g0['expiry']==expiry].copy()
        fit=_fit_one(g)
        if fit is None: continue
        g['sabr_fair_iv']=[sabr_iv(fit['F'],k,fit['T'],fit['alpha'],fit['beta'],fit['rho'],fit['nu']) for k in g['strike'].astype(float)]
        g['surface_residual']=g['implied_volatility'].astype(float)-g['sabr_fair_iv'].astype(float)
        g['surface_z']=_robust_z(g['surface_residual'])
        g['sabr_alpha']=fit['alpha']; g['sabr_beta']=fit['beta']; g['sabr_rho']=fit['rho']; g['sabr_nu']=fit['nu']; g['sabr_rmse']=fit['rmse']
        scored.append(g)
        cal_rows.append({'snapshot_date':g['snapshot_date'].iloc[0],'ticker':ticker,'expiry':expiry,'dte':int(g['dte'].median()),'sabr_alpha':fit['alpha'],'sabr_beta':fit['beta'],'sabr_rho':fit['rho'],'sabr_nu':fit['nu'],'sabr_rmse':fit['rmse'],'sabr_status':'OK'})
    return pd.DataFrame(cal_rows), (pd.concat(scored,ignore_index=True) if scored else pd.DataFrame())
