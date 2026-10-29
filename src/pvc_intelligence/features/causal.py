import numpy as np
import pandas as pd


def features(panel):
    cols = {}
    for col in panel:
        s = panel[col].where(panel[col] > 0)
        for lag in [1, 2, 4, 8, 13]:
            cols[f"{col}_r{lag}"] = np.log(s / s.shift(lag))
        cols[f"{col}_vol8"] = np.log(s / s.shift(1)).rolling(8, min_periods=4).std()
        cols[f"{col}_missing"] = s.isna().astype(float)
    # Differences of log CHANGES; never subtract incompatible quote levels.
    for col in ["futures", "pvc_shanghai", "pvc_sg3_dostyk", "naphtha_cif"]:
        cols[f"{col}_divergence4"] = cols[f"{col}_r4"] - cols["k_r4"]
    return pd.DataFrame(cols, index=panel.index).replace([np.inf, -np.inf], np.nan)


def warning_states(panel):
    changes = panel[["naphtha_cif", "pvc_shanghai", "futures"]].div(
        panel[["naphtha_cif", "pvc_shanghai", "futures"]].shift(4)).sub(1)
    up, down = changes.ge(.10).any(axis=1), changes.le(-.10).any(axis=1)
    ua = (up & ~up.shift(1, fill_value=False)).rolling(4, min_periods=1).max().astype(bool)
    da = (down & ~down.shift(1, fill_value=False)).rolling(4, min_periods=1).max().astype(bool)
    return pd.Series(np.select([ua & da, ua, da], ["BOTH", "UP", "DOWN"], default="NONE"), index=panel.index)
