"""Validate forecast/strategy files against the supplied organizer examples.

The examples establish column names and dates, not future target observations.
No confidence level or exact competition scoring interpolation is inferred.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FORECAST_COLUMNS = ['week_start', 'horizon', 'forecast']
INTERVAL_COLUMNS = ['forecast_lower', 'forecast_upper']
STRATEGY_COLUMNS = ['week_start', 'position']


def _dates(frame, cutoff):
    if len(frame) != 13 or 'week_start' not in frame:
        raise ValueError('Exactly 13 forecast weeks and week_start are required')
    dates = pd.to_datetime(frame.week_start, errors='raise')
    expected = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(weeks=1), periods=13, freq='7D')
    if not pd.DatetimeIndex(dates).equals(expected):
        raise ValueError('Dates must be the next 13 consecutive weeks in order')


def validate_forecast(frame, cutoff, *, require_intervals=False):
    if not set(FORECAST_COLUMNS) <= set(frame):
        raise ValueError('Forecast columns week_start,horizon,forecast are required')
    _dates(frame, cutoff)
    horizons = pd.to_numeric(frame.horizon, errors='raise').to_numpy()
    if not np.array_equal(horizons, np.arange(1, 14)):
        raise ValueError('horizon must be exactly 1 through 13 in order')
    values = pd.to_numeric(frame.forecast, errors='raise').to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Forecast values must be positive and finite')
    present = [c in frame for c in INTERVAL_COLUMNS]
    if any(present) and not all(present):
        raise ValueError('Both interval columns must be supplied together')
    if require_intervals and not all(present):
        raise ValueError('Interval columns are required for this validation mode')
    if all(present):
        lower = pd.to_numeric(frame.forecast_lower, errors='raise').to_numpy()
        upper = pd.to_numeric(frame.forecast_upper, errors='raise').to_numpy()
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError('Interval bounds must be finite; do not invent missing bounds')
        if (lower <= 0).any() or (lower > values).any() or (values > upper).any():
            raise ValueError('Positive interval bounds must contain the point forecast')
    allowed = FORECAST_COLUMNS + (INTERVAL_COLUMNS if all(present) else [])
    if list(frame.columns) != allowed:
        raise ValueError('Use the example column order without extra columns')


def validate_strategy(frame, cutoff):
    if list(frame.columns) != STRATEGY_COLUMNS:
        raise ValueError('Strategy columns must be week_start,position')
    _dates(frame, cutoff)
    positions = pd.to_numeric(frame.position, errors='raise').to_numpy()
    if not np.isfinite(positions).all() or not np.isin(positions, [-1, 0, 1]).all():
        raise ValueError('Positions must be -1,0,+1')


def export_pair(forecast, strategy, cutoff, directory, *, require_intervals=False):
    """Write a new pair, using semicolon and UTF-8 BOM as in supplied examples."""
    validate_forecast(forecast, cutoff, require_intervals=require_intervals)
    validate_strategy(strategy, cutoff)
    directory = Path(directory)
    paths = (directory / 'forecast.csv', directory / 'strategy.csv')
    if any(p.exists() for p in paths):
        raise FileExistsError('Submission export does not overwrite an existing pair')
    directory.mkdir(parents=True, exist_ok=True)
    for frame, path in zip([forecast, strategy], paths):
        out = frame.copy()
        out['week_start'] = pd.to_datetime(out.week_start).dt.strftime('%Y-%m-%d')
        out.to_csv(path, sep=';', encoding='utf-8-sig', index=False)
    return paths
