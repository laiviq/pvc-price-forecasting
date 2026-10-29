"""First upward/downward shocks share a risk set and compete for first event."""
import numpy as np
import pandas as pd
from pvc_intelligence.models.hazard import first_jump_pairs,matured_hazard_data

def first_cause_pairs(target,max_horizon=13,threshold=.08):
    pairs=first_jump_pairs(target,max_horizon,threshold)
    now=target.reindex(pd.DatetimeIndex(pairs.label_date)).to_numpy()
    previous=target.reindex(pd.DatetimeIndex(pairs.label_date)-pd.Timedelta(weeks=1)).to_numpy()
    pairs['cause']=np.where(pairs.event.to_numpy()==0,0,np.where(now>previous,1,2)).astype(int)
    return pairs

def matured_competing_data(features,pairs,cutoff):
    x,_,meta=matured_hazard_data(features,pairs,cutoff)
    return x,meta.cause.to_numpy(dtype=int),meta

def cumulative_incidence(interval_probabilities):
    p=np.asarray(interval_probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p):raise ValueError('Expected nonempty [none,up,down] interval probability rows')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.allclose(p.sum(axis=1),1.,atol=1e-10,rtol=0):raise ValueError('Interval probabilities must be finite and sum to one')
    survival=1.;up=0.;down=0.;rows=[]
    for none,pu,pd_ in p:
        up+=survival*pu;down+=survival*pd_;survival*=none;rows.append([survival,up,down])
    result=np.array(rows);np.testing.assert_allclose(result.sum(axis=1),1.,atol=1e-10,rtol=0)
    return result
