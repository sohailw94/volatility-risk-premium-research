# region imports
from AlgorithmImports import *
# endregion
import pandas as pd

MIN_SCORE_GAP = 1.50


def build_cross_asset_vol_spreads(vrp: pd.DataFrame) -> pd.DataFrame:
    if vrp is None or vrp.empty:
        return pd.DataFrame()
    df=vrp.copy()
    rich=df[df["relative_vol_bucket"]=="RICH_VOL"].copy()
    cheap=df[df["relative_vol_bucket"]=="CHEAP_VOL"].copy()
    # Preserve source ordering: rich/cheap are selected before the ratio-filtered df is made.
    df=df[df["variance_premium_ratio"].between(0.20,5.0)].copy()
    pairs=[]
    for _,r in rich.iterrows():
        for _,c in cheap.iterrows():
            if r["ticker"]==c["ticker"]: continue
            spread_score=r["cross_sectional_vrp_score"]-c["cross_sectional_vrp_score"]
            if spread_score<MIN_SCORE_GAP: continue
            pairs.append({
                "short_vol_ticker":r["ticker"],"long_vol_ticker":c["ticker"],
                "short_vol_vrp_score":r["cross_sectional_vrp_score"],"long_vol_vrp_score":c["cross_sectional_vrp_score"],
                "spread_score":spread_score,"short_vol_ratio":r["variance_premium_ratio"],"long_vol_ratio":c["variance_premium_ratio"],
                "trade_expression":f"SHORT {r['ticker']} VOL / LONG {c['ticker']} VOL",
            })
    out=pd.DataFrame(pairs)
    return out.sort_values("spread_score",ascending=False).reset_index(drop=True) if not out.empty else out

