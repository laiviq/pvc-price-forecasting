"""Prequential source forecasts for delivered-price models.

Future source snapshots can be labels only after their issue timestamps mature.
The second stage consumes the forecast actually issued at each past origin.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def source_features(logs):
    x = pd.DataFrame(index=logs.index)
    for name in ['cny', 'fx']:
        z = logs[name]
        for lag in [1, 4, 13]:
            x[f'{name}_r{lag}'] = z - z.shift(lag)
        x[f'{name}_gap13'] = z - z.rolling(13, min_periods=8).mean()
        x[f'{name}_gap52'] = z - z.rolling(52, min_periods=26).mean()
        x[f'{name}_vol13'] = z.diff().rolling(13, min_periods=8).std()
    phase = 2 * np.pi * (logs.index.dayofyear.to_numpy() - 1) / 365.25
    x['issue_sin'] = np.sin(phase)
    x['issue_cos'] = np.cos(phase)
    return x


def horizon_features(x, origins, horizons):
    origins = pd.DatetimeIndex(origins)
    horizons = np.asarray(horizons, dtype=int)
    if len(origins) != len(horizons) or not ((horizons >= 1) & (horizons <= 13)).all():
        raise ValueError('Each origin needs one horizon in1..13')
    q = x.reindex(origins).reset_index(drop=True)
    q['horizon_fraction'] = horizons / 13
    endpoint = origins + pd.to_timedelta(horizons * 7, unit='D')
    phase = 2 * np.pi * (endpoint.dayofyear.to_numpy() - 1) / 365.25
    q['endpoint_sin'] = np.sin(phase)
    q['endpoint_cos'] = np.cos(phase)
    for name in ['cny', 'fx']:
        for lag in [1, 4, 13]:
            q[f'{name}_r{lag}_h'] = q[f'{name}_r{lag}'] * horizons / 13
    return q


def factor_teachers(logs, x, series, cutoff, window=260):
    rows = []
    for horizon in range(1, 14):
        endpoint = logs.index + pd.Timedelta(weeks=horizon)
        change = logs[series].shift(-horizon) - logs[series]
        valid = (endpoint < cutoff) & (logs.index >= cutoff-pd.Timedelta(weeks=window))
        valid &= logs[['cny', 'fx']].notna().all(axis=1) & change.notna()
        selected = logs.index[valid]
        rows.append(pd.DataFrame({'origin': selected, 'horizon': horizon,
                                  'label_date': endpoint[valid], 'change': change.loc[valid].to_numpy()}))
    meta = pd.concat(rows, ignore_index=True)
    return horizon_features(x, meta.origin, meta.horizon), meta.change.to_numpy(), meta


def factor_forecast(logs, x, cutoff, series):
    X, y, meta = factor_teachers(logs, x, series, cutoff)
    ready = logs.loc[cutoff, ['cny', 'fx']].notna().all() and meta.origin.nunique() >= 104 and len(meta) >= 500
    audit = dict(ready=bool(ready), pairs=len(meta), origins=int(meta.origin.nunique()),
                 last_label=None if meta.empty else meta.label_date.max())
    if not ready:
        return np.zeros(13), audit
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), Ridge(alpha=30.))
    model.fit(X, y)
    query = horizon_features(x, [cutoff]*13, np.arange(1, 14))
    raw = model.predict(query)
    audit['clipped'] = int((abs(raw) > .6).sum())
    return np.clip(raw, -.6, .6), audit


def attach_issued(X, origins, horizons, issued, method):
    """Exact historical key join: no nearest match or later forecast substitution."""
    key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(origins), np.asarray(horizons, dtype=int)],
                                    names=['week_date', 'horizon'])
    q = issued.reindex(key)
    if len(q) != len(X) or q['source_valid'].isna().any():
        raise ValueError('Missing historically issued forecast')
    result = X.copy()
    for name in ['source_valid', 'stage1_ready', 'futures_covered']:
        result['cascade_' + name] = q[name].to_numpy(dtype=float)
    if method == 'Flat':
        a = b = np.zeros(len(q))
    elif method == 'Futures':
        a = q.futures_premium.fillna(0.).to_numpy()
        b = np.zeros(len(q))
    elif method == 'Ridge':
        a, b = q.cny_change.to_numpy(), q.fx_change.to_numpy()
    else:
        raise ValueError(method)
    result['cascade_CNY_change'] = a
    result['cascade_FX_change'] = b
    result['cascade_USD_change'] = a-b
    return result


def past_joint_error_band(issued, logs, cutoff, horizon):
    q = issued.xs(horizon, level='horizon').copy()
    endpoint = q.index + pd.Timedelta(weeks=horizon)
    current_cost = logs.cny-logs.fx
    truth = current_cost.reindex(endpoint).to_numpy()-current_cost.reindex(q.index).to_numpy()
    error = truth - (q.cny_change-q.fx_change).to_numpy()
    valid = (endpoint < cutoff) & (endpoint >= cutoff-pd.Timedelta(weeks=104))
    valid &= q.stage1_ready.to_numpy(dtype=bool) & np.isfinite(error)
    errors = error[valid]
    if len(errors) < 26:
        return (0., 0.), int(len(errors)), None
    return tuple(np.quantile(errors, [.2, .8])), int(len(errors)), endpoint[valid].max()
