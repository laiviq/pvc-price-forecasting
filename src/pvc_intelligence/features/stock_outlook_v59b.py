"""Dimensionless stock-state and issuer-expectation features by report vintage."""
import hashlib
import json

import numpy as np
import pandas as pd

FAMILY = '沿海库存|工厂库存|multiplier1'
CURRENT = ['total_rel12', 'coast_rel12', 'total_change1', 'coast_change1']
FORWARD = ['near_total_ratio', 'far_total_ratio', 'far_coast_ratio', 'total_forward_slope30', 'forward_curve_change']
CONTENT = CURRENT + FORWARD
POLICIES = ['Issue', 'Delay1', 'MetadataFloor']


def next_friday_strict(date):
    date = pd.Timestamp(date).normalize()
    days = (4 - date.weekday()) % 7
    return date + pd.Timedelta(days=days or 7)


def naive_day(value):
    date = pd.Timestamp(value)
    return date.tz_localize(None).normalize() if date.tz is not None else date.normalize()


def snapshots(rows, sources):
    lookup = {s['sha256']: s for s in sources}
    frame = rows[rows.screen_pass & rows.measurement_family.eq(FAMILY)].copy()
    frame['period_end'] = pd.to_datetime(frame.period_end)
    result = []
    for sha, part in frame.groupby('source_sha256'):
        source = lookup[sha]
        issue = naive_day(source['claimed_issue'])
        card = source.get('public_card') or {}
        dated = [issue]
        for value in [card.get('date'), source.get('creation_date')]:
            if value:
                dated.append(naive_day(value))
        base = max(dated)
        if base > issue + pd.Timedelta(days=7):
            continue
        history = part[part.scenario.eq('reported_history_candidate') & part.period_end.le(issue)]
        future = part[part.scenario.eq('issuer_forecast_candidate') & part.period_end.gt(issue) & part.red_stock_cells.ge(2)]
        if history.period_end.nunique() < 6:
            continue
        selected = part[['period_end', 'coastal_stock', 'other_stock_component', 'total_stock', 'scenario']].drop_duplicates().sort_values('period_end')
        fingerprint = hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest()
        floor = max(base, naive_day(card['updateTime'])) if card.get('updateTime') else base
        result.append(dict(source_sha256=sha, issue=issue,
            available={'Issue': next_friday_strict(base), 'Delay1': next_friday_strict(base) + pd.Timedelta(weeks=1),
                       'MetadataFloor': next_friday_strict(floor)},
            history=history.drop_duplicates('period_end').sort_values('period_end'),
            future=future.drop_duplicates('period_end').sort_values('period_end'), fingerprint=fingerprint))
    # Content date reflects earliest known issue of this stock table, not a new
    # assessment when a later weekly report reuses it unchanged.
    first = {}
    for item in result:
        key = item['fingerprint']
        first[key] = min(first.get(key, item['issue']), item['issue'])
    for item in result:
        item['content_date'] = first[item['fingerprint']]
    return sorted(result, key=lambda s: (s['issue'], s['source_sha256']))


def values_at(snapshot, origin):
    issue = snapshot['issue']
    hist = snapshot['history']
    last = hist.iloc[-1]
    if (origin - issue).days > 42 or (origin - last.period_end).days > 70:
        return None
    if (last[['total_stock', 'coastal_stock']] <= 0).any():
        return None
    twelve = hist.tail(12)
    if (twelve[['total_stock', 'coastal_stock']] <= 0).any().any():
        return None
    result = {c: np.nan for c in CONTENT}
    result['total_rel12'] = float(np.log(last.total_stock / twelve.total_stock.median()))
    result['coast_rel12'] = float(np.log(last.coastal_stock / twelve.coastal_stock.median()))
    if len(hist) >= 2:
        previous = hist.iloc[-2]
        if last.period_end.to_period('M').ordinal - previous.period_end.to_period('M').ordinal == 1:
            result['total_change1'] = float(np.log(last.total_stock / previous.total_stock))
            result['coast_change1'] = float(np.log(last.coastal_stock / previous.coastal_stock))
    future = snapshot['future']
    future = future[future.period_end.gt(origin) & future.total_stock.gt(0) & future.coastal_stock.gt(0)]
    first_days, last_days = np.nan, np.nan
    if len(future):
        near, far = future.iloc[0], future.iloc[-1]
        result['near_total_ratio'] = float(np.log(near.total_stock / last.total_stock))
        result['far_total_ratio'] = float(np.log(far.total_stock / last.total_stock))
        result['far_coast_ratio'] = float(np.log(far.coastal_stock / last.coastal_stock))
        result['total_forward_slope30'] = result['far_total_ratio'] * 30 / (far.period_end - last.period_end).days
        result['forward_curve_change'] = float(np.log(far.total_stock / near.total_stock))
        first_days = (near.period_end - origin).days
        last_days = (far.period_end - origin).days
    result.update(available=1., source_age_days=float((origin - issue).days),
        content_age_days=float((origin - snapshot['content_date']).days),
        history_age_days=float((origin - last.period_end).days), history_months=float(len(hist)),
        future_months=float(len(future)), first_future_days=first_days, last_future_days=last_days,
        historical_span_days=float((hist.iloc[-1].period_end - hist.iloc[0].period_end).days))
    return result


def feature_frame(rows, sources, origins, policy):
    if policy not in POLICIES:
        raise ValueError(policy)
    items = snapshots(rows, sources)
    records, audit = [], []
    for origin in pd.DatetimeIndex(origins):
        known = [s for s in items if s['available'][policy] <= origin]
        known.sort(key=lambda s: (s['available'][policy], s['issue'], s['source_sha256']))
        value, selected = None, None
        for candidate in reversed(known):
            candidate = dict(candidate, content_date=min(s['issue'] for s in known
                if s['fingerprint'] == candidate['fingerprint']))
            value = values_at(candidate, origin)
            if value is not None:
                selected = candidate
                break
        if value is None:
            value = {c: np.nan for c in CONTENT + ['source_age_days', 'content_age_days', 'history_age_days',
                'history_months', 'future_months', 'first_future_days', 'last_future_days', 'historical_span_days']}
            value['available'] = 0.
        value['known_issue_count'] = float(len({s['issue'] for s in known}))
        for name in CONTENT:
            value['missing_' + name] = float(pd.isna(value[name]))
        records.append(value)
        audit.append(dict(week_date=origin, policy=policy,
            source_sha256=selected['source_sha256'] if selected else None,
            claimed_issue=selected['issue'] if selected else pd.NaT,
            available_date=selected['available'][policy] if selected else pd.NaT,
            current_present=any(pd.notna(value[c]) for c in CURRENT),
            forward_present=any(pd.notna(value[c]) for c in FORWARD)))
    frame = pd.DataFrame(records, index=pd.DatetimeIndex(origins, name='week_date'))
    assert len(frame.columns) == 28
    return frame.add_prefix('s59_'), pd.DataFrame(audit)
