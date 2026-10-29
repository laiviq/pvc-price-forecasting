"""Explicit post-quote information contract; legacy strict helpers stay intact."""
import numpy as np
import pandas as pd


def price_training(features, target, cutoff, inclusive=False):
    cutoff = pd.Timestamp(cutoff)
    pieces, values, metadata = [], [], []
    for horizon in range(1, 14):
        outcome = np.log(target.shift(-horizon) / target)
        dates = target.index + pd.Timedelta(weeks=horizon)
        known = dates <= cutoff if inclusive else dates < cutoff
        mask = known & target.notna() & outcome.notna()
        piece = features.loc[mask].copy()
        piece['horizon'] = horizon
        pieces.append(piece)
        values.append(outcome.loc[mask].to_numpy())
        metadata.append(pd.DataFrame(dict(origin=target.index[mask], horizon=horizon,
                                          label_date=dates[mask])))
    return pd.concat(pieces), np.concatenate(values), pd.concat(metadata, ignore_index=True)


def risk_training(features, pairs, cutoff, inclusive=False):
    known = pairs.label_date.le(pd.Timestamp(cutoff)) if inclusive else pairs.label_date.lt(pd.Timestamp(cutoff))
    meta = pairs.loc[known].reset_index(drop=True)
    matrix = features.reindex(pd.DatetimeIndex(meta.origin)).reset_index(drop=True)
    matrix['horizon'] = meta.horizon.to_numpy()
    return matrix, meta.event.to_numpy(dtype=int), meta
