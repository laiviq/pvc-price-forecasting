import pandas as pd
import numpy as np
from pvc_intelligence.features.world import ecm_residual

def features(raw,k,delay=0):
 r=raw.copy();r['quote_date']=pd.to_datetime(r.quote_date);r['available_at']=r.quote_date+pd.Timedelta(days=1,weeks=delay)
 r=r[r.pvc_spot.gt(0)].sort_values('available_at')
 m=pd.merge_asof(pd.DataFrame({'week_date':k.index}),r,left_on='week_date',right_on='available_at',direction='backward')
 valid=(m.week_date-m.quote_date).dt.days.le(21)
 cols=['pvc_spot','nearby_settlement','main_settlement','methanol_spot','coke_spot','caustic_soda_spot']
 vals=m[cols].where(valid,axis=0).set_axis(k.index);x={}
 v=vals.pvc_spot
 x['spot_log']=np.log(v);x['spot_rel52']=np.log(v)-np.log(v).rolling(52,min_periods=26).median();x['spot_ecm']=ecm_residual(k,v)
 for lag in [1,4,13]:x['spot_r'+str(lag)]=np.log(v/v.shift(lag))
 for side in ['nearby','main']:
  a=np.log(v/vals[side+'_settlement']);x[side+'_basis_log']=a;x[side+'_basis_change4']=a-a.shift(4)
 x['spot_missing']=v.isna().astype(float)
 for c in ['methanol_spot','coke_spot','caustic_soda_spot']:
  v=vals[c]
  for lag in [4,13]:x[c+'_r'+str(lag)]=np.log(v/v.shift(lag))
  x[c+'_rel52']=np.log(v)-np.log(v).rolling(52,min_periods=26).median()
  x[c+'_missing']=v.isna().astype(float)
 lineage=m.loc[valid,['week_date','quote_date','available_at','source_url']]
 return pd.DataFrame(x,index=k.index),lineage
