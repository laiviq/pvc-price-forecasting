"""Hold source and transforms fixed while removing selected NaOH observations."""
import pandas as pd
from pvc_intelligence.features.caustic_credit_v65 import build_features, RAW, INTERACTIONS

CONTENT = ['caustic65_' + c for c in RAW + INTERACTIONS]


def restricted_content(raw, observed_dates, calendar, delay_days=0):
    allow = pd.DatetimeIndex(observed_dates).normalize()
    dates = pd.to_datetime(raw['quote_date'])
    subset = raw[raw['product'].ne('caustic') | dates.isin(allow)].copy()
    frames, audit = build_features(subset, calendar, delay_days)
    return {name: frames[name][CONTENT] for name in ['Raw', 'Credit']}, audit


def replace_content(long_frame, content):
    if list(content.columns) != CONTENT or not content.index.equals(long_frame.index):
        raise ValueError('Unmatched schema or calendar')
    result = long_frame.copy()
    result[CONTENT] = content
    return result
