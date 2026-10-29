"""A fixed2x2 experiment for original market and external physical blocks."""
import numpy as np


def joint_blocks(base, original, physical):
    if not base.index.equals(original.index) or not base.index.equals(physical.index):
        raise ValueError('Source calendars must match exactly')
    if any(frame.columns.duplicated().any() for frame in [base,original,physical]):
        raise ValueError('Duplicate source columns')
    if set(base).intersection(original) or set(base).intersection(physical) or set(original).intersection(physical):
        raise ValueError('Already included source column must be removed before joining')
    original_numeric=[c for c in original if not c.endswith('_missing')]
    physical_numeric=[c for c in physical if not c.endswith(('_missing','_age_days'))]
    combined=base.join(original).join(physical)
    result={}
    for name in ['Control','Original','Physical','Joint']:
        x=combined.copy()
        if name not in ['Original','Joint']:x[original_numeric]=np.nan
        if name not in ['Physical','Joint']:x[physical_numeric]=np.nan
        result[name]=x
    return result,dict(original_numeric=original_numeric,physical_numeric=physical_numeric,
                       original_controls=[c for c in original if c not in original_numeric],
                       physical_controls=[c for c in physical if c not in physical_numeric])
