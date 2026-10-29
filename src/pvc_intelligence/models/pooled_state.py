"""Shared conditional price distributions, integrating hypothetical future states."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pvc_intelligence.models.directional_mixture import quantile_model,finite_state_decision


def state_columns(X, states):
    states=np.asarray(states,dtype=int)
    if len(states)!=len(X) or not np.isin(states,[0,1,2]).all():
        raise ValueError('Expected one hypothetical state0/1/2 per row')
    q=X.copy();q['scenario_first_up']=(states==1).astype(float);q['scenario_first_down']=(states==2).astype(float)
    return q


def hypothetical_queries(query):
    """No target argument: future states are enumerated, never observed."""
    return pd.concat([state_columns(query,np.full(len(query),s,dtype=int)) for s in [0,1,2]],ignore_index=True)


def mature_state_prior(meta,horizons=range(1,14)):
    counts=pd.crosstab(meta.horizon,meta.state).reindex(index=list(horizons),columns=[0,1,2],fill_value=0).fillna(0.)
    return ((counts+1.).div(counts.sum(axis=1)+3.,axis=0)).to_numpy()


def pooled_predict(X,y,meta,query,cif,parameters,minimum=100):
    """True state is allowed only in matured teacher meta; queries have no truth."""
    X0=state_columns(X,np.zeros(len(X),dtype=int));q0=state_columns(query,np.zeros(len(query),dtype=int))
    if len(y)<minimum:
        model=CatBoostRegressor(**parameters,verbose=False,allow_writing_files=False).fit(X0,y)
        point=np.clip(model.predict(q0),-.6,.6)
        return {k:point.copy() for k in ['Unconditional','Prior','CIF']},dict(total_sample_fallback=True,uq=np.full((len(query),7),np.nan),conditional=np.full((len(query),3,7),np.nan),prior=mature_state_prior(meta,query.horizon))
    uq=np.asarray(quantile_model(parameters).fit(X0,y).predict(q0))
    uncond,ud=finite_state_decision(np.ones((len(query),1)),uq[:,None,:])
    conditional_model=quantile_model(parameters).fit(state_columns(X,meta.state),y)
    queried=np.asarray(conditional_model.predict(hypothetical_queries(query)))
    conditional=queried.reshape(3,len(query),7).transpose(1,0,2)
    prior=mature_state_prior(meta,query.horizon)
    prior_point,pd_=finite_state_decision(prior,conditional)
    cif_point,cd=finite_state_decision(cif,conditional)
    return dict(Unconditional=uncond,Prior=prior_point,CIF=cif_point),dict(total_sample_fallback=False,uq=ud['sorted_quantiles'][:,0,:],conditional=cd['sorted_quantiles'],prior=prior,unconditional_crossing=ud['quantile_crossing'][:,0],conditional_crossing=cd['quantile_crossing'])
