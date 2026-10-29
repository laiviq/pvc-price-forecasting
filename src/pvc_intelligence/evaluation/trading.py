"""Frozen forecast-path positions and the DEMO gross P&L convention.

P&L and drawdown are in the target's index units, without costs or notional.
These long/short scoring positions are not an inventory sales policy.
"""
import numpy as np


def _vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite vector")
    return result


def forecast_positions(last_known, forecast, *, deadband=0.0):
    """Decide the whole path at origin, using adjacent predicted prices only.

    Week 1 compares forecast[0] to last_known; later weeks compare adjacent
    forecasts. deadband is an absolute index change, fixed before evaluation.
    """
    predicted = _vector(forecast, "forecast")
    if not np.isfinite(last_known) or not np.isfinite(deadband) or deadband < 0:
        raise ValueError("last_known and nonnegative deadband must be finite")
    changes = np.diff(np.r_[last_known, predicted])
    return np.where(np.abs(changes) > deadband, np.sign(changes), 0).astype(int)


def evaluate_positions(last_known, future_actual, positions):
    """Require observed consecutive weeks; never fill future target labels."""
    actual = _vector(future_actual, "future_actual")
    pos = _vector(positions, "positions")
    if not np.isfinite(last_known) or actual.shape != pos.shape:
        raise ValueError("Finite entry price and matching path lengths required")
    if not np.isin(pos, [-1, 0, 1]).all():
        raise ValueError("Positions must be -1, 0 or +1")
    changes = np.diff(np.r_[last_known, actual])
    weekly = pos * changes
    equity = np.r_[0.0, np.cumsum(weekly)]
    drawdown = np.maximum.accumulate(equity) - equity
    previous = np.r_[0.0, pos[:-1]]
    return {
        "pnl_index": float(weekly.sum()),
        "max_drawdown_index": float(drawdown.max()),
        "active_weeks": int(np.count_nonzero(pos)),
        "turnover_units": float(np.abs(pos - previous).sum()),
        "weekly_pnl": weekly,
    }
