from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
import qc_config_v2 as cfg
from sabr_v2 import sabr_iv

MAX_ACCEPTED_RMSE=cfg.SABR_MAX_ACCEPTED_RMSE
MIN_CONFIDENCE=cfg.SABR_MIN_CONFIDENCE

def safe_sabr(row,strike_mult):
    F=float(row["forward"]);K=F*strike_mult;T=max(float(row["dte"])/365.0,1/365)
    return sabr_iv(F,K,T,row["sabr_alpha"],row["sabr_beta"],row["sabr_rho"],row["sabr_nu"])

def build_features(df):
    rows=[]
    for _,row in df.iterrows():
        atm=safe_sabr(row,1.00);p95=safe_sabr(row,.95);c105=safe_sabr(row,1.05);p90=safe_sabr(row,.90);c110=safe_sabr(row,1.10)
        if not np.all(np.isfinite([atm,p95,c105,p90,c110])):continue
        rows.append({**row.to_dict(),"sabr_atm_iv":atm,"sabr_put_95_iv":p95,"sabr_call_105_iv":c105,"sabr_put_90_iv":p90,"sabr_call_110_iv":c110,"sabr_skew_95_105":p95-c105,"sabr_downside_skew":p95-atm,"sabr_upside_skew":c105-atm,"sabr_smile_curvature":.5*(p95+c105)-atm,"sabr_wing_richness":.5*(p90+c110)-atm})
    return pd.DataFrame(rows)

def sabr_confidence(row):
    rmse=row.get("sabr_rmse",np.nan)
    if pd.isna(rmse) or row.get("sabr_status")!="OK":return 0.0
    conf=float(np.exp(-8.0*rmse))
    if abs(float(row.get("fitted_atm_iv",np.nan))-float(row.get("observed_atm_iv",np.nan)))>.08:conf*=.50
    return float(np.clip(conf,0,1))

def add_term_structure_features(features):
    if features.empty:return features
    pieces=[]
    for _,g in features.groupby("ticker"):
        g=g.sort_values("dte").copy();front=g.iloc[0];back=g.iloc[-1]
        g["sabr_term_slope"]=(back["sabr_atm_iv"]-front["sabr_atm_iv"])/max(back["dte"]-front["dte"],1);g["sabr_front_atm_iv"]=front["sabr_atm_iv"];g["sabr_back_atm_iv"]=back["sabr_atm_iv"];pieces.append(g)
    return pd.concat(pieces,ignore_index=True)

def add_cross_sectional_zscores(features):
    features=features.copy();cols=["sabr_atm_iv","sabr_skew_95_105","sabr_smile_curvature","sabr_wing_richness","sabr_term_slope","sabr_rmse"]
    for c in cols:
        mu,sd=features[c].mean(),features[c].std();features[f"{c}_z"]=0.0 if pd.isna(sd) or sd==0 else (features[c]-mu)/sd
    features["sabr_surface_richness_score"]=(.35*features.sabr_atm_iv_z+.20*features.sabr_wing_richness_z+.20*features.sabr_smile_curvature_z+.15*features.sabr_skew_95_105_z-.10*features.sabr_rmse_z)
    return features

def classify_surface(row):
    if row.get("sabr_confidence",0.0)<MIN_CONFIDENCE:return "SABR_LOW_CONFIDENCE"
    score=row.get("sabr_surface_richness_score_adj",0.0);curv=row.get("sabr_smile_curvature_z",0.0);skew=row.get("sabr_skew_95_105_z",0.0)
    if score>=1:return "SURFACE_RICH"
    if score<=-1:return "SURFACE_CHEAP"
    if curv>=1:return "WINGS_RICH"
    if curv<=-1:return "WINGS_CHEAP"
    if skew>=1:return "PUT_SKEW_RICH"
    if skew<=-1:return "CALL_SKEW_RICH"
    return "SURFACE_FAIR"

def build_sabr_surface_features(sabr,vrp):
    if sabr is None or sabr.empty:return pd.DataFrame()
    s=sabr.copy();valid=(s.sabr_status.eq("OK")&s.sabr_fit_quality.isin(["EXCELLENT","GOOD","FAIR"])&s.sabr_rmse.le(MAX_ACCEPTED_RMSE)&~s.sabr_hit_alpha_bound.fillna(False)&~s.sabr_hit_rho_bound.fillna(False)&~s.sabr_hit_nu_bound.fillna(False));s=s[valid].copy()
    f=build_features(s)
    if f.empty:return f
    f=add_term_structure_features(f);f=add_cross_sectional_zscores(f)
    if vrp is not None and not vrp.empty:
        keep=[c for c in ["ticker","variance_premium_ratio","cross_sectional_vrp_score","relative_vol_bucket","forecast_vol_20d","atm_iv_20d"] if c in vrp]
        f=f.merge(vrp[keep].drop_duplicates("ticker"),on="ticker",how="left")
    f["sabr_confidence"]=f.apply(sabr_confidence,axis=1);f["sabr_surface_richness_score_adj"]=f.sabr_surface_richness_score*f.sabr_confidence;f["sabr_surface_bucket"]=f.apply(classify_surface,axis=1)
    return f.sort_values("sabr_surface_richness_score_adj",ascending=False).reset_index(drop=True)


