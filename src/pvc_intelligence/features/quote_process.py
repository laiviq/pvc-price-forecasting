"""Observed price persistence; no inference that unchanged quotes are stale."""
from collections import deque

import numpy as np
import pandas as pd

CONTENT = ['equal_run', 'zero_share4', 'zero_share8', 'zero_share13', 'zero_share26',
           'last_nonzero_return', 'abs_last_nonzero_return', 'mean_abs_nonzero13',
           'last3_change_sum', 'last_change_sign', 'zero_share4minus26']


def quote_process_features(target):
    if not isinstance(target.index, pd.DatetimeIndex) or not target.index.is_unique or not target.index.is_monotonic_increasing:
        raise ValueError('Expected an ordered unique weekly calendar')
    if len(target) > 1 and not target.index.to_series().diff().dropna().eq(pd.Timedelta(weeks=1)).all():
        raise ValueError('Missing calendar weeks must be explicit NaN rows')
    if target.dropna().le(0).any():
        raise ValueError('Observed target values must be positive')
    previous = target.shift()
    adjacent = target.notna() & previous.notna()
    equal = target.eq(previous).astype(float).where(adjacent)
    change = np.log(target / previous).where(adjacent)
    f = pd.DataFrame(index=target.index)
    runs, last, sums = [], [], []
    history = deque()
    run = 0
    for i, (known, zero, value) in enumerate(zip(adjacent, equal, change)):
        if not known:
            run = 0
            history.clear()
        else:
            run = min(run + 1, 52) if zero == 1 else 0
            if value != 0:
                history.append((i, value))
        while history and i - history[0][0] > 52:
            history.popleft()
        runs.append(float(run) if pd.notna(target.iloc[i]) else np.nan)
        last.append(history[-1][1] if history else np.nan)
        sums.append(sum(v for _, v in list(history)[-3:]) if len(history) >= 3 else np.nan)
    f['equal_run'] = runs
    for window in [4, 8, 13, 26]:
        f[f'zero_share{window}'] = equal.rolling(window, min_periods=(window + 1) // 2).mean()
    f['last_nonzero_return'] = last
    f['abs_last_nonzero_return'] = f.last_nonzero_return.abs()
    f['mean_abs_nonzero13'] = change.where(change.ne(0)).abs().rolling(13, min_periods=2).mean()
    f['last3_change_sum'] = sums
    f['last_change_sign'] = np.sign(f.last_nonzero_return)
    f['zero_share4minus26'] = f.zero_share4 - f.zero_share26
    f['valid_adjacent_fraction26'] = adjacent.astype(float).rolling(26, min_periods=13).mean()
    f['adjacent_missing'] = (~adjacent).astype(float)
    f['day_of_month'] = target.index.day.astype(float)
    f['days_to_month_end'] = (target.index.days_in_month - target.index.day).astype(float)
    f['month_end_within7days'] = f.days_to_month_end.lt(7).astype(float)
    for name in CONTENT:
        f['missing_' + name] = f[name].isna().astype(float)
    assert len(f.columns) == 27
    return f.add_prefix('qp58_')
