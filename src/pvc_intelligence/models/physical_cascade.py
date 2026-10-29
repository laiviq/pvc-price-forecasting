"""Forecast a cost-proxy residual around an already-known futures anchor."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pvc_intelligence.models.cascade import factor_teachers,horizon_features


def anchor_query(X, origins, horizons, issued_curve):
    key=pd.MultiIndex.from_arrays([pd.DatetimeIndex(origins),np.asarray(horizons,dtype=int)],names=['week_date','horizon'])
    known=issued_curve.reindex(key)
    if known.source_valid.isna().any():raise ValueError('Missing issued curve key')
    result=X.copy()
    premium=known.futures_premium.fillna(0.).to_numpy()
    result['curve_premium']=premium
    result['curve_covered']=known.futures_covered.to_numpy(dtype=float)
    return result,premium


def cost_forecast(logs, features, curve, cutoff, parameters):
    X,y,meta=factor_teachers(logs,features,'cost',cutoff)
    X,anchor=anchor_query(X,meta.origin,meta.horizon,curve)
    query=horizon_features(features,[cutoff]*13,range(1,14))
    query,current=anchor_query(query,[cutoff]*13,range(1,14),curve)
    ready=logs.loc[cutoff,['cny','fx']].notna().all() and meta.origin.nunique()>=104 and len(meta)>=500
    audit=dict(ready=bool(ready),pairs=len(meta),origins=int(meta.origin.nunique()),last_label=None if meta.empty else meta.label_date.max())
    if not ready:return np.clip(current,-.6,.6),audit
    model=CatBoostRegressor(**parameters,verbose=False,allow_writing_files=False).fit(X,y-anchor)
    raw=model.predict(query)+current
    audit['clipped']=int((abs(raw)>.6).sum())
    return np.clip(raw,-.6,.6),audit


def payload(X, origins, horizons, issued, method):
    key=pd.MultiIndex.from_arrays([pd.DatetimeIndex(origins),np.asarray(horizons,dtype=int)],names=['week_date','horizon'])
    q=issued.reindex(key)
    if q.source_valid.isna().any():raise ValueError('Missing historical forecast')
    result=X.copy()
    for name in ['source_valid','stage1_ready','futures_covered']:
        result['cascade70_'+name]=q[name].to_numpy(dtype=float)
    if method=='Flat':values=np.zeros(len(q))
    elif method=='Futures':values=q.futures_premium.fillna(0.).to_numpy()
    elif method in ['Core','Physical']:values=q[method+'_cost_change'].to_numpy()
    else:raise ValueError(method)
    result['cascade70_cost_change']=values
    return result


def error_band(issued, logs, cutoff, horizon, method):
    q=issued.xs(horizon,level='horizon');endpoint=q.index+pd.Timedelta(weeks=horizon)
    truth=logs.cost.reindex(endpoint).to_numpy()-logs.cost.reindex(q.index).to_numpy()
    error=truth-q[method+'_cost_change'].to_numpy()
    valid=(endpoint<cutoff)&(endpoint>=cutoff-pd.Timedelta(weeks=104))&q.stage1_ready.to_numpy(dtype=bool)&np.isfinite(error)
    if valid.sum()<26:return (0.,0.),int(valid.sum()),None
    return tuple(np.quantile(error[valid],[.2,.8])),int(valid.sum()),endpoint[valid].max()
