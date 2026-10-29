"""Past-only target state, regional market features and public daily snapshots."""
import hashlib
import json

import numpy as np
import pandas as pd


def target_state(k):
    log=np.log(k)
    out=pd.DataFrame(index=k.index)
    out['state_log_k']=log
    out['state_rel52']=log-log.rolling(52,min_periods=26).median()
    out['state_drawdown13']=log-log.rolling(13,min_periods=8).max()
    out['state_range13']=log.rolling(13,min_periods=8).max()-log.rolling(13,min_periods=8).min()
    return out


def ecm_residual(k,market,window=104,min_pairs=52):
    """Rolling log-index relationship; intercept removes arbitrary index bases.

    Fits use only paired data at/before each feature date. This is a statistical
    disequilibrium feature, never a USD/t spread or a proven cointegrating law.
    """
    y=np.log(k);x=np.log(market);valid=x.notna()&y.notna()
    x=x.where(valid);y=y.where(valid)
    meanx=x.rolling(window,min_periods=min_pairs).mean()
    meany=y.rolling(window,min_periods=min_pairs).mean()
    varx=x.rolling(window,min_periods=min_pairs).var()
    vary=y.rolling(window,min_periods=min_pairs).var()
    cov=x.rolling(window,min_periods=min_pairs).cov(y)
    beta=cov/(varx+1e-4)
    alpha=meany-beta*meanx
    residual=np.log(k)-alpha-beta*np.log(market)
    sd=np.sqrt((vary+beta*beta*varx-2*beta*cov).clip(lower=1e-4))
    return (residual/sd).clip(-10,10)


def workbook_markets(raw,k,*,delay_weeks=1):
    meta=[c for c in raw.columns if c not in ['report_date','k','выборка']]
    features={};catalog=[];lineage=[]
    target_name='ПВХ SG-5 (Китай) DAP Достык, USD/т'
    raw=raw.loc[raw.prod_version.ne(target_name)].copy()
    for identity,part in raw.groupby(meta,dropna=False,sort=True):
        props=dict(zip(meta,identity));part=part.sort_values('report_date').dropna(subset=['k'])
        part=part.loc[part.k.gt(0)]
        if len(part)<52:continue
        sid='wb_'+hashlib.sha256(json.dumps(props,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()[:10]
        monthly='месяч' in str(props.get('frequency','')) or 'month' in str(props.get('prod_version','')).lower()
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


def daily_public_features(records,calendar,*,extra_delay_weeks=0):
    frames={};lineage=[]
    r=records.copy()
    for c in ['quote_date','available_at']:r[c]=pd.to_datetime(r[c])
    r['available_at']=r.available_at+pd.Timedelta(weeks=extra_delay_weeks)
    for sid,p in r.groupby('series',sort=True):
        p=p.sort_values(['available_at','quote_date']).drop_duplicates('available_at',keep='last')
        matched=pd.merge_asof(pd.DataFrame({'week_date':pd.DatetimeIndex(calendar)}),p[['quote_date','available_at','value']],left_on='week_date',right_on='available_at',direction='backward')
        age=(matched.week_date-matched.quote_date).dt.days
        cap=21 if sid.startswith('cbr') else 14
        value=pd.Series(matched.value.where(age.le(cap)).to_numpy(),index=calendar)
        frames[sid+'_level']=np.log1p(value)
        for lag in [1,4,13]:frames[sid+f'_change{lag}']=np.log1p(value)-np.log1p(value.shift(lag))
        frames[sid+'_rel52']=np.log1p(value)-np.log1p(value).rolling(52,min_periods=26).median()
        frames[sid+'_missing']=value.isna().astype(float)
        for row in matched.loc[matched.value.notna()&age.le(cap)].itertuples():lineage.append({'week_date':row.week_date,'series':sid,'quote_date':row.quote_date,'available_at':row.available_at})
    return pd.DataFrame(frames,index=calendar),pd.DataFrame(lineage)
