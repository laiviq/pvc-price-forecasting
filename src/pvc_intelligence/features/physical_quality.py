"""Derived current-state indicators require a usable current level."""
import numpy as np

def require_current_physical_level(frame):
    result=frame.copy();changes={}
    for column in frame:
        if not column.endswith('_level'):continue
        prefix=column[:-len('level')];missing=frame[column].isna()
        for suffix in ['d4','d13','rank52']:
            derived=prefix+suffix
            if derived not in frame:continue
            count=int((missing&frame[derived].notna()).sum())
            if count:changes[derived]=count
            if suffix=='rank52':
                def rank(values):
                    if not np.isfinite(values[-1]):return np.nan
                    observed=values[np.isfinite(values)]
                    return float((observed<=values[-1]).mean())
                corrected=frame[column].rolling(52,min_periods=13).apply(rank,raw=True).where(~missing)
                changed=~(np.isclose(frame[derived],corrected,equal_nan=True))
                if int(changed.sum()):changes[derived+'_recomputed']=int(changed.sum())
                result[derived]=corrected
            result.loc[missing,derived]=np.nan
            indicator=derived+'_missing'
            if indicator in result:result[indicator]=result[derived].isna().astype(float)
    return result,changes
