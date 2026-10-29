"""Scheduled policy information known at the forecast origin, not later events."""
import pandas as pd

def scheduled_policy_features(events,cutoffs,horizons=range(1,14)):
    e=events.copy()
    for c in ['available_at','effective_at']:e[c]=pd.to_datetime(e[c])
    rows=[]
    for cutoff in pd.DatetimeIndex(cutoffs):
        known=e.loc[e.available_at.le(cutoff)]
        for h in horizons:
            future=cutoff+pd.Timedelta(weeks=h)
            pending=known.loc[known.effective_at.gt(cutoff)&known.effective_at.le(future)]
            nextdates=known.loc[known.effective_at.gt(cutoff),'effective_at']
            rows.append({'cutoff':cutoff,'horizon':h,'policy_due_within_horizon':len(pending),
                         'policy_already_effective':int(known.effective_at.le(cutoff).sum()),
                         'weeks_to_next_announced_change':(nextdates.min()-cutoff).days/7 if len(nextdates) else float('nan')})
    return pd.DataFrame(rows)
