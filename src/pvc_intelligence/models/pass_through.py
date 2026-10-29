"""Bounded weekly adjustment to an observed cost proxy, in log index units.

The statistical basis is not a measured transport bill or proven cointegration.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class AdjustmentFit:
    coefficients: np.ndarray
    center_y: float
    center_x: float
    labels: pd.DatetimeIndex
    cost_lag: int
    target_only: bool
    objective: float


def fit_adjustment(target, log_cost, cutoff, cost_lag=0, *, target_only=False,
                   window=104, min_pairs=52):
    """Use only already known weekly transition outcomes, without filling gaps."""
    if cost_lag < 0:
        raise ValueError('Cost lag must not be negative')
    if not target.index.equals(log_cost.index):
        raise ValueError('Target and cost must have the same weekly index')
    y = np.log(target.where(target.gt(0)))
    dy = y.diff()
    x = log_cost.shift(cost_lag)
    frame = pd.DataFrame({'dy': dy, 'previous_dy': dy.shift(),
                          'previous_y': y.shift(), 'cost': x})
    mask = (frame.index < cutoff) & (frame.index >= cutoff - pd.Timedelta(weeks=window))
    columns = ['dy', 'previous_dy'] if target_only else list(frame)
    frame = frame.loc[mask].dropna(subset=columns)
    if len(frame) < min_pairs:
        return None
    cy = float(frame.previous_y.mean()) if not target_only else 0.
    cx = float(frame.cost.mean()) if not target_only else 0.
    if target_only:
        matrix = np.column_stack([np.ones(len(frame)), frame.previous_dy])
        initial = [0., .1]
        bounds = [(None, None), (0., .95)]
        constraints = ()
    else:
        matrix = np.column_stack([np.ones(len(frame)), -(frame.previous_y - cy),
                                  frame.cost - cx, frame.previous_dy])
        initial = [0., .25, .125, .1]
        bounds = [(None, None), (0., 1.), (0., 1.), (0., .95)]
        constraints = ({'type': 'ineq', 'fun': lambda b: b[1] - b[2],
                        'jac': lambda b: np.array([0., 1., -1., 0.])},)
    outcome = frame.dy.to_numpy()

    def objective(b):
        residual = matrix @ b - outcome
        absolute = abs(residual)
        loss = np.where(absolute <= .02, .5 * residual ** 2,
                        .02 * (absolute - .01)).mean()
        grad = matrix.T @ np.clip(residual, -.02, .02) / len(outcome)
        loss += 1e-6 * np.dot(b[1:], b[1:])
        grad[1:] += 2e-6 * b[1:]
        return float(loss), grad

    result = minimize(objective, initial, jac=True, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'ftol': 1e-12, 'maxiter': 2000})
    if not result.success:
        raise RuntimeError(f'Adjustment optimization failed: {result.message}')
    if not target_only and result.x[2] > result.x[1] + 1e-9:
        raise RuntimeError('Pass-through constraint was not satisfied')
    return AdjustmentFit(result.x, cy, cx, pd.DatetimeIndex(frame.index),
                         cost_lag, target_only, float(result.fun))


def forecast_adjustment(fit, target, log_cost, cutoff, *, premium=None, horizon=13):
    """Freeze the cost scenario on issue date; later source rows are ignored."""
    if pd.isna(target.get(cutoff, np.nan)):
        raise ValueError('The issue-date target must be observed')
    current_y = float(np.log(target.loc[cutoff]))
    if fit is None:
        return np.repeat(target.loc[cutoff], horizon), {'fallback': True, 'clipped': 0}
    previous = target.get(cutoff - pd.Timedelta(weeks=1), np.nan)
    dy = float(current_y - np.log(previous)) if previous > 0 else 0.
    current_x = log_cost.get(cutoff, np.nan)
    queue = []
    if not fit.target_only:
        for h in range(1, horizon + 1):
            cost_horizon = h - fit.cost_lag
            if cost_horizon <= 0:
                value = log_cost.get(cutoff + pd.Timedelta(weeks=cost_horizon), np.nan)
            else:
                value = current_x
                if premium is not None and np.isfinite(premium.get(cost_horizon, np.nan)):
                    value += premium[cost_horizon]
            queue.append(value)
        if not np.isfinite(current_x) or not np.isfinite(queue).all():
            return np.repeat(target.loc[cutoff], horizon), {'fallback': True, 'clipped': 0}
    y = current_y
    predictions = []
    clipped = 0
    for h in range(horizon):
        if fit.target_only:
            change = fit.coefficients[0] + fit.coefficients[1] * dy
        else:
            c, kappa, theta, phi = fit.coefficients
            change = c - kappa * (y - fit.center_y) + theta * (queue[h] - fit.center_x) + phi * dy
        raw = y + change
        y = float(np.clip(raw, current_y - .6, current_y + .6))
        clipped += int(raw != y)
        dy = y - (current_y if not predictions else np.log(predictions[-1]))
        predictions.append(float(np.exp(y)))
    return np.asarray(predictions), {'fallback': False, 'clipped': clipped}
