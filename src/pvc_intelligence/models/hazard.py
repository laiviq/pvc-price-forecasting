"""Observed discrete-time first-jump histories with explicit right censoring."""
import numpy as np
import pandas as pd

def first_jump_pairs(target, max_horizon=13, threshold=.08):
    """One row per observed event-free interval, stopping at event or missing pair.

    A missing weekly quote ends a path. It does not manufacture a negative label
    and does not permit a later jump to be called the first event on that path.
    """
    if not isinstance(target.index,pd.DatetimeIndex) or not target.index.is_unique or not target.index.is_monotonic_increasing:
        raise ValueError('Expected a unique ordered weekly calendar')
    if len(target)>1 and not (target.index.to_series().diff().dropna()==pd.Timedelta(weeks=1)).all():
        raise ValueError('Missing calendar rows must be represented by explicit NaN quotes')
    if target.dropna().le(0).any():
        raise ValueError('Observed prices must be positive')
    rows=[]
    for i, origin in enumerate(target.index):
        if pd.isna(target.iloc[i]) or target.iloc[i] <= 0:
            continue
        for h in range(1, max_horizon+1):
            j=i+h
            if j>=len(target):
                break
            previous,current=target.iloc[j-1],target.iloc[j]
            if pd.isna(previous) or pd.isna(current) or previous<=0 or current<=0:
                break
            event=bool(abs(current/previous-1)>=threshold)
            rows.append({'origin':origin,'horizon':h,'label_date':target.index[j],
                         'event':int(event)})
            if event:
                break
    return pd.DataFrame(rows,columns=['origin','horizon','label_date','event'])

def matured_hazard_data(features, pairs, cutoff):
    known=pairs.loc[pairs.label_date.lt(pd.Timestamp(cutoff))].reset_index(drop=True)
    x=features.reindex(pd.DatetimeIndex(known.origin)).reset_index(drop=True)
    x['horizon']=known.horizon.to_numpy()
    return x,known.event.to_numpy(dtype=int),known

def path_probability(hazards):
    hazards=np.asarray(hazards,dtype=float)
    if not np.isfinite(hazards).all() or ((hazards<0)|(hazards>1)).any():
        raise ValueError('Hazards must be finite probabilities')
    return float(-np.expm1(np.log1p(-np.clip(hazards,0,1-1e-15)).sum())) if not (hazards==1).any() else 1.
