"""Monthly calibration inside the report known at the forecasting origin."""
import numpy as np
import pandas as pd

from pvc_intelligence.features.stock_outlook_v59b import naive_day, next_friday_strict, POLICIES


def month_price_date(month_end):
    date = naive_day(month_end)
    return date + pd.Timedelta(days=(4 - date.weekday()) % 7)


def report_vintages(rows, sources):
    lookup = {s['sha256']: s for s in sources}
    frame = rows[rows.screen_pass].copy()
    frame['period_end'] = pd.to_datetime(frame.period_end)
    result = []
    for sha, z in frame.groupby('source_sha256'):
        source = lookup[sha]
        issue = naive_day(source['claimed_issue'])
        card = source.get('public_card') or {}
        base = max([issue] + [naive_day(v) for v in [card.get('date'), source.get('creation_date')] if v])
        if base > issue + pd.Timedelta(days=7):
            continue
        history = z[z.scenario.eq('reported_history_candidate') & z.period_end.le(issue)
                    & z.total_stock.gt(0)].drop_duplicates('period_end').sort_values('period_end')
        forward = z[z.scenario.eq('issuer_forecast_candidate') & z.red_stock_cells.ge(2)
                    & z.period_end.gt(issue) & z.total_stock.gt(0)].drop_duplicates('period_end').sort_values('period_end')
        if len(history) < 13 or not len(forward):
            continue
        floor = max(base, naive_day(card['updateTime'])) if card.get('updateTime') else base
        result.append(dict(source_sha256=sha, issue=issue, family=z.measurement_family.iloc[0],
            history=history, forward=forward,
            available={'Issue': next_friday_strict(base), 'Delay1': next_friday_strict(base) + pd.Timedelta(weeks=1),
                       'MetadataFloor': next_friday_strict(floor)}))
    return sorted(result, key=lambda x: (x['issue'], x['source_sha256']))


def select_report(reports, origin, policy):
    if policy not in POLICIES:
        raise ValueError(policy)
    known = [r for r in reports if r['available'][policy] <= origin
             and (origin - r['issue']).days <= 42
             and (origin - r['history'].period_end.iloc[-1]).days <= 70]
    return max(known, key=lambda r: (r['available'][policy], r['issue'], r['source_sha256'])) if known else None


def monthly_teachers(report, target, reference_lookup, cutoff):
    records = []
    history = list(report['history'].itertuples())
    for previous, current in zip(history[:-1], history[1:]):
        if current.period_end.to_period('M').ordinal - previous.period_end.to_period('M').ordinal != 1:
            continue
        start, end = month_price_date(previous.period_end), month_price_date(current.period_end)
        if end >= cutoff:
            continue
        days = (end - start).days
        horizon = days // 7
        if days <= 0 or days % 7 or not 1 <= horizon <= 13:
            continue
        initial, actual = target.get(start, np.nan), target.get(end, np.nan)
        prediction = reference_lookup.get((start, horizon), np.nan)
        if not np.isfinite([initial, actual, prediction]).all() or min(initial, actual, prediction) <= 0:
            continue
        records.append(dict(stock_month=current.period_end, origin=start, label_date=end, horizon=horizon,
            x=float(np.log(current.total_stock / previous.total_stock) * 30 / days),
            y=float(np.log(actual / prediction) * 30 / days), actual=float(actual),
            past_oof_prediction=float(prediction)))
    return pd.DataFrame(records, columns=['stock_month', 'origin', 'label_date', 'horizon', 'x', 'y', 'actual', 'past_oof_prediction'])


def fit_calibration(teachers):
    if len(teachers) < 12:
        return None
    x = teachers.x.to_numpy(dtype=float)
    y = teachers.y.to_numpy(dtype=float)
    xm, ym = float(x.mean()), float(y.mean())
    raw_beta = float(np.sum((x - xm) * (y - ym)) / (np.sum((x - xm) ** 2) + 1.0))
    beta = float(np.clip(raw_beta, -1., 0.))
    return dict(n=len(x), xmean=xm, ymean=ym, beta=beta, raw_beta=raw_beta,
                intercept=ym - beta * xm, last_label=teachers.label_date.max())


def guidance_delta(report, origin, forecast_dates):
    # Stocks remain reported history or issuer expectations. Mapping their
    # month ends to price Fridays is the fixed monthly alignment assumption.
    last = report['history'].iloc[-1]
    knots = [(month_price_date(last.period_end), float(last.total_stock))]
    knots += [(month_price_date(r.period_end), float(r.total_stock)) for r in report['forward'].itertuples()]
    knots = sorted(knots)
    dates = pd.DatetimeIndex([p[0] for p in knots])
    logs = np.log([p[1] for p in knots])
    result = np.full(len(forecast_dates), np.nan)
    if not dates.is_unique or origin < dates[0] or origin > dates[-1]:
        return result
    numeric = (dates - dates[0]).days.to_numpy(dtype=float)
    at_origin = float(np.interp((origin - dates[0]).days, numeric, logs))
    for idx, date in enumerate(pd.DatetimeIndex(forecast_dates)):
        if dates[0] <= date <= dates[-1]:
            result[idx] = float(np.interp((date - dates[0]).days, numeric, logs) - at_origin)
    return result


def adjust_path(base_prediction, current_target, origin, report, calibration):
    base = np.asarray(base_prediction, dtype=float)
    if base.shape != (13,) or not np.isfinite(base).all() or (base <= 0).any() or current_target <= 0:
        raise ValueError('Expected thirteen finite positive reference points and a positive origin quote')
    control, stock = base.copy(), base.copy()
    delta = np.full(13, np.nan)
    active = np.zeros(13, dtype=bool)
    if report is None or calibration is None:
        return control, stock, active, delta
    forecast_dates = pd.date_range(origin + pd.Timedelta(weeks=1), periods=13, freq='7D')
    delta = guidance_delta(report, origin, forecast_dates)
    active = np.isfinite(delta)
    days = np.arange(1, 14) * 7
    common = .5 * calibration['ymean'] * days / 30
    inventory = .5 * calibration['beta'] * (delta - calibration['xmean'] * days / 30)
    logbase = np.log(base / current_target)
    control[active] = current_target * np.exp(np.clip((logbase + common)[active], -.6, .6))
    stock[active] = current_target * np.exp(np.clip((logbase + common + inventory)[active], -.6, .6))
    return control, stock, active, delta
