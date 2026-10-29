"""Issue-dated CPC forecasts of temperature terciles, never realised weather."""
import io
import numpy as np
import pandas as pd

STATIONS = {'Houston': '72243', 'PointComfort': '72255',
            'LakeCharles': '72240', 'BatonRouge': '72232'}
PROBABILITIES = ['TBLW', 'TNRM', 'TABV', 'PBLW', 'PNRM', 'PABV']


def parse_cpc(text, issue_date, kind):
    if kind not in ['fcst610', 'fcst814']:
        raise ValueError('Unknown forecast period')
    lines = text.splitlines()
    if not lines or lines[0].strip() != 'PARM=TBLW;TNRM;TABV;TCAT;PBLW;PNRM;PABV;PCAT':
        raise ValueError('Not a CPC probability table')
    frame = pd.read_csv(io.StringIO('\n'.join(lines[1:])), sep=r'\s+',
                        dtype={'STN': str, 'YYMMDD/HHMM': str})
    expected = ['STN', 'YYMMDD/HHMM', 'TBLW', 'TNRM', 'TABV', 'TCAT', 'PBLW', 'PNRM', 'PABV', 'PCAT']
    if list(frame) != expected or frame.empty or frame.isna().any().any():
        raise ValueError('Unexpected schema or missing probability')
    if frame.STN.duplicated().any() or not frame.STN.str.fullmatch(r'\d{5}').all():
        raise ValueError('Ambiguous station identifier')
    issue = pd.Timestamp(issue_date).normalize()
    start, end, middle = (6, 10, 8) if kind == 'fcst610' else (8, 14, 11)
    midpoint = issue + pd.Timedelta(days=middle)
    if not frame['YYMMDD/HHMM'].eq(midpoint.strftime('%y%m%d/0000')).all():
        raise ValueError('Forecast midpoint is inconsistent with issue date')
    values = frame[PROBABILITIES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise ValueError('Probability outside 0..100 percent')
    for group in [['TBLW', 'TNRM', 'TABV'], ['PBLW', 'PNRM', 'PABV']]:
        if not np.allclose(frame[group].sum(axis=1), 100, rtol=0, atol=.021):
            raise ValueError('Tercile probabilities do not sum to 100')
    if not frame.TCAT.isin([1, 2, 3]).all() or not frame.PCAT.isin([1, 2, 3]).all():
        raise ValueError('Unknown tercile category')
    frame['issue_date'] = issue
    frame['valid_from'] = issue + pd.Timedelta(days=start)
    frame['valid_to'] = issue + pd.Timedelta(days=end)
    frame['kind'] = kind
    return frame


def outlook_at(rows, cutoff, delay_days=0):
    """Return a still-valid published outlook for each lead; absence stays absent."""
    cutoff = pd.Timestamp(cutoff).normalize()
    result = {}
    for kind in ['fcst610', 'fcst814']:
        eligible = rows[rows.kind.eq(kind) &
                        (rows.issue_date + pd.Timedelta(days=1 + delay_days)).le(cutoff) &
                        rows.valid_to.ge(cutoff)]
        if eligible.empty:
            result[kind] = None
            continue
        issue = eligible.issue_date.max()
        selected = eligible[eligible.issue_date.eq(issue)].copy()
        if selected.STN.duplicated().any():
            raise ValueError('Multiple source values for one station and issue')
        if set(selected.STN) != set(STATIONS.values()):
            result[kind] = None
        else:
            result[kind] = selected.set_index('STN')
    return result
