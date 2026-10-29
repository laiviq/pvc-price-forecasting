"""Correct explicit-frequency interpretation; legacy frozen panels remain unchanged."""
import hashlib,json,re
import numpy as np
import pandas as pd
from pvc_intelligence.features.world import ecm_residual

def is_monthly_market(props):
    frequency=str(props.get('frequency','')).lower()
    product=str(props.get('prod_version','')).lower()
    # An explicit monthly-average label describes the measurement period, even
    # when a vendor refreshes the row every day. "month 1 weekly avg" describes
    # contract tenor and must not be treated as a monthly measurement.
    if re.search(r'\bmonthly(?:\s+average|\s+avg)?\b|\bmonth\s+(?:average|avg)\b',product):return True
    if 'месяч' in frequency or frequency in ['monthly','month']:return True
    if 'ежеднев' in frequency or 'еженедель' in frequency or frequency in ['daily','weekly']:return False
    return bool(re.search(r'\bmonthly\b|\bmonth\s+avg\b',product))

def workbook_markets_explicit_frequency(raw,k,*,delay_weeks=1):
    meta=[c for c in raw.columns if c not in ['report_date','k','выборка']]
    features={};catalog=[];lineage=[]
    target_name='ПВХ SG-5 (Китай) DAP Достык, USD/т'
    raw=raw.loc[raw.prod_version.ne(target_name)].copy()
    for identity,part in raw.groupby(meta,dropna=False,sort=True):
        props=dict(zip(meta,identity));part=part.sort_values('report_date').dropna(subset=['k'])
        part=part.loc[part.k.gt(0)]
        if len(part)<52:continue
        sid='wb_'+hashlib.sha256(json.dumps(props,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()[:10]
        monthly=is_monthly_market(props)
        base=part.report_date+pd.offsets.MonthEnd(0) if monthly else part.report_date
        days=(4-base.dt.day_of_week)%7
        part=part.assign(available_at=base+pd.to_timedelta(days,unit='D')+pd.Timedelta(weeks=delay_weeks))
        part=part.sort_values(['available_at','report_date']).drop_duplicates('available_at',keep='last')
        if part.report_date.duplicated().any():raise ValueError('Ambiguous market identity')
        matched=pd.merge_asof(pd.DataFrame({'week_date':k.index}),part[['available_at','report_date','k']].rename(columns={'k':'market_k'}),
            left_on='week_date',right_on='available_at',direction='backward')
        maxage=65 if monthly else 21
        age=(matched.week_date-matched.available_at).dt.days
        values=pd.Series(matched.market_k.where(age.le(maxage)).to_numpy(),index=k.index)
        product=str(props.get('prod_version',''));country=str(props.get('country',''))
        is_pvc='pvc' in product.lower() or 'ПВХ' in product
        group='feedstock' if not is_pvc else ('russia_corridor' if country in ['Россия','Казахстан'] else ('china' if country=='Китай' else 'world_pvc'))
        columns=[]
        for lag in [1,4,13]:
            name=sid+f'_r{lag}';features[name]=np.log(values/values.shift(lag));columns.append(name)
        name=sid+'_rel52';features[name]=np.log(values)-np.log(values).rolling(52,min_periods=26).median();columns.append(name)
        name=sid+'_ecm';features[name]=ecm_residual(k,values);columns.append(name)
        features[sid+'_missing']=values.isna().astype(float)
        catalog.append({'series':sid,'group':group,**props,'raw_observations':len(part),'monthly_end_gate':monthly,
                        'features':json.dumps(columns),'availability_verified':False})
        used=matched.loc[matched.market_k.notna()&age.le(maxage)]
        for row in used.itertuples():lineage.append({'week_date':row.week_date,'series':sid,'quote_date':row.report_date,
            'available_at':row.available_at,'availability_verified':False,'delay_weeks_assumed':delay_weeks})
    return pd.DataFrame(features,index=k.index),pd.DataFrame(catalog),pd.DataFrame(lineage)


