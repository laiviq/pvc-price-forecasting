"""Zero-atom/nonzero-quantile approximation with an sMAPE point decision."""
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

QUANTILES = np.array([.05, .20, .35, .50, .65, .80, .95])
QUADRATURE_WEIGHTS = np.array([.125, .15, .15, .15, .15, .15, .125])


def finite_smape_decision(probability_zero, conditional_quantiles):
    p = np.asarray(probability_zero, dtype=float).reshape(-1)
    q = np.asarray(conditional_quantiles, dtype=float)
    if q.shape != (len(p), 7) or not np.isfinite(q).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite probabilities and seven conditional quantiles')
    crossing = (np.diff(q, axis=1) < 0).any(axis=1)
    values = np.concatenate([np.zeros((len(p), 1)), np.sort(np.clip(q, -.6, .6), axis=1)], axis=1)
    weights = np.concatenate([p[:, None], (1 - p[:, None]) * QUADRATURE_WEIGHTS[None, :]], axis=1)
    # Positive prices: sMAPE(exp(r),exp(s)) =200*tanh(abs(r-s)/2).
    losses = 200 * np.tanh(np.abs(values[:, :, None] - values[:, None, :]) / 2)
    expected = (losses * weights[:, None, :]).sum(axis=2)
    selected = expected.argmin(axis=1)  # Zero atom is deterministic first tie-break.
    return values[np.arange(len(p)), selected], dict(p0=p, quantiles=values[:, 1:],
        quantile_crossing=crossing, expected_smape=expected[np.arange(len(p)), selected])


def fit_predict(kind, x, y, query, parameters):
    params = dict(parameters, verbose=False, allow_writing_files=False)
    if kind == 'MAE':
        prediction = CatBoostRegressor(**params).fit(x, y).predict(query)
        return np.clip(prediction, -.6, .6), dict(fallback=False)
    if kind != 'Mixture':
        raise ValueError(kind)
    flat = np.asarray(y) == 0
    if len(np.unique(flat)) < 2 or (~flat).sum() < 100:
        prediction, _ = fit_predict('MAE', x, y, query, parameters)
        return prediction, dict(fallback=True)
    params['loss_function'] = 'Logloss'
    classifier = CatBoostClassifier(**params).fit(x, flat.astype(int))
    p = classifier.predict_proba(query)[:, 1]
    params['loss_function'] = 'MultiQuantile:alpha=' + ','.join(map(str, QUANTILES))
    conditional = CatBoostRegressor(**params).fit(x.loc[~flat], np.asarray(y)[~flat])
    quantiles = np.asarray(conditional.predict(query))
    prediction, diagnostics = finite_smape_decision(p, quantiles)
    diagnostics['fallback'] = False
    return prediction, diagnostics
