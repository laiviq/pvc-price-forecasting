import numpy as np


def smape(actual, predicted):
    a, p = np.broadcast_arrays(np.asarray(actual, float), np.asarray(predicted, float))
    den = np.abs(a) + np.abs(p)
    return np.divide(200 * np.abs(a - p), den, out=np.zeros_like(den), where=den != 0)


def block_bootstrap_delta(values, block=13, repetitions=2000, seed=1729):
    """Moving blocks of origin-level paired errors, preserving 13-week overlap.

    Descriptive uncertainty, not a correction for historical model selection.
    """
    v = np.asarray(values, float)
    n = len(v)
    if n < block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, (repetitions, int(np.ceil(n / block))))
    samples = v[(starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, :n]].mean(axis=1)
    return tuple(np.quantile(samples, [.025, .975]))
