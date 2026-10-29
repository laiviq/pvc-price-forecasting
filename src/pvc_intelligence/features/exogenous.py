"""Strict released-data feature builder, no access to future outcomes."""
import numpy as np
import pandas as pd

from pvc_intelligence.data.pit import CUTOFF, asof_snapshot
from pvc_intelligence.models.research import TARGET_FEATURES

CATEGORIES = ["china_trade", "global_trade", "geopolitical_sanctions", "energy_policy",
              "china_energy_supply", "rail_logistics", "sea_logistics", "pandemic"]


def target_features(k):
    d = {f"k_r{lag}": np.log(k / k.shift(lag)) for lag in [1, 2, 4, 8, 13]}
    d["k_vol8"] = d["k_r1"].rolling(8, min_periods=4).std()
    return pd.DataFrame(d, index=k.index)


def event_features(events, dates, *, delay_weeks=0, exclude_category=None, census_only=False):
    e = events.loc[events.availability_verified.eq(True)].copy()
    if exclude_category:
        e = e.loc[e.category.ne(exclude_category)]
    if census_only:
        e = e.loc[e.event_id.str.startswith("FR_")]
    e["release"] = pd.to_datetime(e.available_at) + pd.Timedelta(weeks=delay_weeks)
    dates=pd.DatetimeIndex(dates)
    # Include observed publications before the first target week in its trailing
    # window. Reindexing to the target calendar before rolling would lose them.
    full_dates=pd.date_range(dates.min()-pd.Timedelta(weeks=12),dates.max(),freq='W-FRI')
    x = pd.DataFrame(index=full_dates)
    for category in CATEGORIES:
        count = e.loc[e.category.eq(category)].groupby("release").size().reindex(x.index, fill_value=0)
        for lookback in [4, 13]:
            x[f"event_{category}_{lookback}w"] = count.rolling(lookback, min_periods=1).sum()
    # Activity counts, not a signed tariff-rate or causal geopolitical index.
    trade = e.loc[e.category.isin(["china_trade", "global_trade"])]
    for action in ["tightening", "easing"]:
        count = trade.loc[trade.policy_action.eq(action)].groupby("release").size().reindex(x.index, fill_value=0)
        x[f"trade_{action}_4w"] = count.rolling(4, min_periods=1).sum()
    return x.reindex(dates)


def released_macro(records, dates, *, delay_weeks=0):
    r = records.loc[records.availability_verified.eq(True)].copy()
    r["available_at"] = pd.to_datetime(r.available_at) + pd.Timedelta(weeks=delay_weeks)
    frames, lineages = [], []
    for sid, part in r.groupby("series"):
        max_age = 75 if sid.startswith('property_') else 45 if 'pmi' in sid else 21
        v, lineage = asof_snapshot(part, dates, verified_only=True, max_age_days=max_age)
        frames.append(v)
        lineages.append(lineage)
    values = pd.concat(frames, axis=1)
    cols, groups = {}, {}
    for sid in values:
        names = []
        s = values[sid]
        unit = r.loc[r.series.eq(sid), 'unit'].unique()
        if len(unit) != 1:
            raise ValueError(f"Incompatible units: {sid}")
        for lag in [1, 4, 13]:
            name = f"{sid}_change{lag}"
            cols[name] = np.log(s/s.shift(lag)) if unit[0] == 'CNY/t' else (s-s.shift(lag)) / 100
            names.append(name)
        if unit[0] != 'CNY/t':
            name = sid + '_level'
            cols[name] = (s-50)/100 if 'pmi' in sid else s/100
            names.append(name)
        cols[sid + '_missing'] = s.isna().astype(float)
        names.append(sid + '_missing')
        groups[sid] = names
    x = pd.DataFrame(cols, index=values.index).replace([np.inf,-np.inf], np.nan)
    # Changes in distinct normalized series, not physical price-level spreads.
    for cost in ['nbs_thermal_coal5500', 'nbs_coke', 'nbs_lng', 'nbs_lpg']:
        if cost in values and 'nbs_sg5' in values:
            name = cost + '_sg5_divergence4'
            x[name] = x[cost+'_change4'] - x.nbs_sg5_change4
            groups[cost].append(name)
    return values, x, pd.concat(lineages, ignore_index=True), groups
