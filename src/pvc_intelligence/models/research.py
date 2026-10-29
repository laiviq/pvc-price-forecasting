import numpy as np
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_FEATURES = [f"k_r{lag}" for lag in [1, 2, 4, 8, 13]] + ["k_vol8"]


def ridge_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=30.0))


def pooled_model(params):
    return CatBoostRegressor(**params, verbose=False, allow_writing_files=False)


def mature_pairs(x, target, cutoff, min_origin=None):
    """All direct horizon pairs whose outcomes were observed strictly before cutoff."""
    pieces, ys, label_dates = [], [], []
    for h in range(1, 14):
        y = np.log(target.shift(-h) / target)
        dates = target.index + np.timedelta64(7 * h, "D")
        mask = (dates < cutoff) & target.notna() & y.notna()
        if min_origin is not None:
            mask &= target.index >= min_origin
        p = x.loc[mask].copy()
        p["horizon"] = h
        pieces.append(p)
        ys.append(y.loc[mask].to_numpy())
        label_dates.extend(dates[mask])
    import pandas as pd
    return pd.concat(pieces), np.concatenate(ys), pd.DatetimeIndex(label_dates)
