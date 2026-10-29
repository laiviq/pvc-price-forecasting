"""Relative NaOH coproduct signals; no concentration or physical margin assumptions."""
import numpy as np
import pandas as pd

PRODUCTS = ('pvc', 'carbide', 'caustic')
RAW = ['caustic_' + s for s in ('r1', 'r4', 'r13', 'rel52')]
INTERACTIONS = ['carbide_caustic4', 'carbide_caustic13', 'pvc_caustic4',
                'pvc_caustic13', 'cost_pressure_weak_caustic4', 'joint_revenue_weak4']
CONFLICT = pd.Timestamp('2024-12-26')


def build_features(raw, calendar, delay_days=0):
    if delay_days not in (0, 7):
        raise ValueError('Unregistered availability policy')
    calendar = pd.DatetimeIndex(calendar, name='week_date')
    source = raw[raw['product'].isin(PRODUCTS)].copy()
    source['quote_date'] = pd.to_datetime(source['quote_date'])
    if source.duplicated(['quote_date', 'product']).any():
        raise ValueError('Duplicate assessment key')
    if not source.value_cny_t.gt(0).all():
        raise ValueError('Nonpositive source quote')
    source = source[source.quote_date.le(pd.Timestamp('2026-07-02'))]
    x = pd.DataFrame(index=calendar)
    audit = pd.DataFrame(index=calendar)
    for product in PRODUCTS:
        q = source[source['product'].eq(product)][['quote_date', 'value_cny_t']].copy()
        # A Friday quote is not known that same Friday under this policy.
        q['available_at'] = q.quote_date + pd.to_timedelta(
            (4 - q.quote_date.dt.dayofweek) % 7, unit='D')
        q.loc[q.available_at.eq(q.quote_date), 'available_at'] += pd.Timedelta(days=7)
        q['available_at'] += pd.Timedelta(days=delay_days)
        if product == 'caustic':
            q.loc[q.quote_date.eq(CONFLICT), 'value_cny_t'] = np.nan
        m = pd.merge_asof(pd.DataFrame({'week_date': calendar}), q.sort_values('available_at'),
                          left_on='week_date', right_on='available_at', direction='backward')
        m.index = calendar
        age = (m.week_date - m.quote_date).dt.days
        valid = age.between(0, 21) & m.value_cny_t.notna()
        price = m.value_cny_t.where(valid)
        x[product + '_valid'] = valid.astype(float)
        x[product + '_age21'] = age.div(21).clip(upper=1).where(m.quote_date.notna())
        log = np.log(price)
        for horizon in (1, 4, 13):
            x[product + '_r' + str(horizon)] = log - log.shift(horizon)
        x[product + '_rel52'] = log - log.rolling(52, min_periods=26).median()
        audit[product + '_quote_date'] = m.quote_date
        audit[product + '_available_at'] = m.available_at
        audit[product + '_valid'] = valid
    for horizon in (4, 13):
        x['carbide_caustic' + str(horizon)] = x['carbide_r' + str(horizon)] - x['caustic_r' + str(horizon)]
        x['pvc_caustic' + str(horizon)] = x['pvc_r' + str(horizon)] - x['caustic_r' + str(horizon)]
    x['cost_pressure_weak_caustic4'] = (x.carbide_r4 - x.pvc_r4).clip(lower=0) * (-x.caustic_r4).clip(lower=0)
    x['joint_revenue_weak4'] = (-x.pvc_r4).clip(lower=0) * (-x.caustic_r4).clip(lower=0)
    assert len(x.columns) == 24
    audit['all_valid'] = audit[[p + '_valid' for p in PRODUCTS]].all(axis=1)
    frames = {}
    for name, masked in [('Control', RAW + INTERACTIONS), ('Raw', INTERACTIONS), ('Credit', [])]:
        f = x.copy()
        f[masked] = np.nan
        f.columns = ['caustic65_' + c for c in f]
        frames[name] = f
    return frames, audit
