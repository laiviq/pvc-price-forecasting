"""First-event state teachers and finite-distribution price decisions."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pvc_intelligence.models.hazard import first_jump_pairs
from pvc_intelligence.models.quote_mixture import QUANTILES,QUADRATURE_WEIGHTS
from pvc_intelligence.models.competing import cumulative_incidence


def factorized_incidence(any_hazards,up_given_event):
    a=np.asarray(any_hazards,dtype=float);d=np.asarray(up_given_event,dtype=float)
    if a.ndim!=1 or d.shape!=a.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(d).all() or ((a<0)|(a>1)|(d<0)|(d>1)).any():
        raise ValueError('Expected finite equal-length any/direction interval probabilities')
    return cumulative_incidence(np.column_stack([1-a,a*d,a*(1-d)]))


def state_price_pairs(target,max_horizon=13,threshold=.08):
    # Reuse the calendar/positive-price validator; do not infer jumps across gaps.
    first_jump_pairs(target,max_horizon,threshold)
    rows=[]
    for i,origin in enumerate(target.index):
        initial=target.iloc[i]
        if pd.isna(initial):continue
        state=0;unknown=False
        for horizon in range(1,max_horizon+1):
            j=i+horizon
            if j>=len(target):break
            now,previous=target.iloc[j],target.iloc[j-1]
            if state==0:
                if pd.isna(now) or pd.isna(previous):unknown=True
                elif not unknown and abs(now/previous-1)>=threshold:state=1 if now>previous else 2
            if pd.notna(now) and (state>0 or not unknown):
                rows.append(dict(origin=origin,horizon=horizon,label_date=target.index[j],state=state,log_return=np.log(now/initial)))
    return pd.DataFrame(rows,columns=['origin','horizon','label_date','state','log_return']).sort_values(['horizon','origin'],kind='stable').reset_index(drop=True)


def mature_state_price_data(features,pairs,cutoff):
    meta=pairs[pairs.label_date.lt(pd.Timestamp(cutoff))].reset_index(drop=True)
    x=features.reindex(pd.DatetimeIndex(meta.origin)).copy();x['horizon']=meta.horizon.to_numpy()
    return x,meta.log_return.to_numpy(),meta


def finite_state_decision(probabilities,quantiles):
    p=np.asarray(probabilities,dtype=float);q=np.asarray(quantiles,dtype=float)
    if p.ndim!=2 or q.ndim!=3 or q.shape[:2]!=p.shape or q.shape[2]!=7 or not len(p):
        raise ValueError('Expected n×states probabilities and n×states×7 quantiles')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or ((p<0)|(p>1)).any() or not np.allclose(p.sum(axis=1),1,atol=1e-10,rtol=0):
        raise ValueError('Invalid distribution')
    crossing=(np.diff(q,axis=2)<0).any(axis=2)
    values=np.sort(np.clip(q,-.6,.6),axis=2).reshape(len(p),-1)
    weights=(p[:,:,None]*QUADRATURE_WEIGHTS[None,None,:]).reshape(len(p),-1)
    loss=200*np.tanh(np.abs(values[:,:,None]-values[:,None,:])/2)
    expected=(loss*weights[:,None,:]).sum(axis=2);selected=expected.argmin(axis=1)
    return values[np.arange(len(p)),selected],dict(quantile_crossing=crossing,expected_smape=expected[np.arange(len(p)),selected],sorted_quantiles=values.reshape(q.shape))


def quantile_model(parameters):
    p=dict(parameters);p['loss_function']='MultiQuantile:alpha='+','.join(map(str,QUANTILES))
    return CatBoostRegressor(**p,verbose=False,allow_writing_files=False)


def price_components(x,y,states,query,probabilities,parameters,minimum=100):
    mae=CatBoostRegressor(**parameters,verbose=False,allow_writing_files=False).fit(x,y)
    base=np.clip(mae.predict(query),-.6,.6);counts=np.bincount(np.asarray(states,dtype=int),minlength=3)
    result=dict(MAE=base,Quantile=base.copy(),State=base.copy())
    detail=dict(state_counts=counts,quantile_fallback=len(y)<minimum,state_fallback=bool((counts<minimum).any()),unconditional_quantiles=np.full((len(query),7),np.nan),state_quantiles=np.full((len(query),3,7),np.nan),unconditional_crossings=np.zeros(len(query),dtype=bool),state_crossings=np.zeros((len(query),3),dtype=bool))
    if len(y)>=minimum:
        uq=np.asarray(quantile_model(parameters).fit(x,y).predict(query));point,d=finite_state_decision(np.ones((len(query),1)),uq[:,None,:])
        result['Quantile']=point;detail.update(unconditional_quantiles=d['sorted_quantiles'][:,0,:],unconditional_crossings=d['quantile_crossing'][:,0])
    if not detail['state_fallback']:
        conditional=[]
        for state in range(3):
            mask=np.asarray(states)==state
            conditional.append(np.asarray(quantile_model(parameters).fit(x.loc[mask],np.asarray(y)[mask]).predict(query)))
        sq=np.stack(conditional,axis=1);point,d=finite_state_decision(probabilities,sq)
        result['State']=point;detail.update(state_quantiles=d['sorted_quantiles'],state_crossings=d['quantile_crossing'])
    return result,detail
