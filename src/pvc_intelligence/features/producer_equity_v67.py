"""Past-only action-neutral equity indices and market-controlled weekly signals."""
import numpy as np
import pandas as pd

STOCKS=('zhongtai','tianye');MARKETS=('shanghai','shenzhen')
RAW_SUFFIXES=('r1','r4','r13','alpha1','alpha4','alpha13','vol4','rel52')
JOINT=('basket_alpha1','basket_alpha4','basket_alpha13','disagreement4')
CONTENT=[s+'_'+c for s in STOCKS for c in RAW_SUFFIXES]+list(JOINT)


def action_neutral_index(raw,actions):
    """Use published ex references, or announced per10share distributions across gaps."""
    r=raw.sort_values('date').copy();r['date']=pd.to_datetime(r.date)
    a=actions.copy();a['announcement_date']=pd.to_datetime(a.announcement_date);a['ex_date']=pd.to_datetime(a.ex_date)
    if a.ex_date.duplicated().any() or not a.announcement_date.le(a.ex_date).all():
        raise ValueError('Ambiguous or backdated action')
    if r.date.duplicated().any() or not r.close.gt(0).all():raise ValueError('Invalid price history')
    values=[];audit=[];level=1.;previous=None;previous_date=None
    for row in r.itertuples():
        if previous is None:reference=np.nan;level=1.;events=a.iloc[:0]
        else:
            events=a[a.ex_date.gt(previous_date)&a.ex_date.le(row.date)]
            reference=previous
            for event in events.sort_values('ex_date').itertuples():
                reference=(reference-event.cash_per10/10)/(1+event.bonus_per10/10+event.transfer_per10/10)
            if pd.notna(getattr(row,'prevclose',np.nan)):reference=float(row.prevclose)
            if reference<=0:raise ValueError('Nonpositive action-neutral reference')
            level*=row.close/reference
        values.append(level);audit.append(dict(date=row.date,reference=reference,actions_since_previous_quote=len(events)))
        previous=float(row.close);previous_date=row.date
    r['adjusted_close']=values
    return r,pd.DataFrame(audit)


def hfq_index(raw,factors):
    r=raw.sort_values('date').copy();f=factors.sort_values('date').copy()
    r['date']=pd.to_datetime(r.date);f['date']=pd.to_datetime(f.date)
    if f.date.duplicated().any() or not f.hfq_factor.gt(0).all():raise ValueError('Invalid dated factors')
    m=pd.merge_asof(r,f,on='date',direction='backward')
    m['adjusted_close']=m.close*m.hfq_factor
    return m


def build_features(series,calendar,delay_days=0):
    if set(series)!=set(STOCKS+MARKETS) or delay_days not in (0,7):raise ValueError('Unregistered source/policy')
    calendar=pd.DatetimeIndex(calendar,name='week_date');x=pd.DataFrame(index=calendar);audit=pd.DataFrame(index=calendar)
    for name in STOCKS+MARKETS:
        q=series[name][['date','adjusted_close']].copy().sort_values('date');q['date']=pd.to_datetime(q.date)
        if q.date.duplicated().any() or not q.adjusted_close.gt(0).all():raise ValueError('Invalid adjusted series')
        daily=np.log(q.adjusted_close/q.adjusted_close.shift())
        daily=daily.where(q.date.diff().dt.days.le(14))
        window=pd.Series(daily.to_numpy(),index=pd.DatetimeIndex(q.date))
        q['vol4']=window.rolling('28D',min_periods=10).std().to_numpy()
        q['count28']=pd.Series(1.,index=pd.DatetimeIndex(q.date)).rolling('28D').sum().to_numpy()
        q['available_at']=q.date+pd.Timedelta(days=1+delay_days)
        m=pd.merge_asof(pd.DataFrame({'week_date':calendar}),q.sort_values('available_at'),left_on='week_date',right_on='available_at',direction='backward');m.index=calendar
        age=(m.week_date-m.date).dt.days;valid=age.between(1,14)
        log=np.log(m.adjusted_close.where(valid))
        x[name+'_valid']=valid.astype(float);x[name+'_age14']=age.div(14).clip(upper=1)
        x[name+'_count28']=m.count28.div(24).clip(upper=1)
        for horizon in (1,4,13):x[name+'_r'+str(horizon)]=log-log.shift(horizon)
        x[name+'_rel52']=log-log.rolling(52,min_periods=26).median()
        x[name+'_vol4']=m.vol4.where(valid)
        audit[name+'_quote_date']=m.date;audit[name+'_available_at']=m.available_at;audit[name+'_valid']=valid
    for stock,market in [('zhongtai','shenzhen'),('tianye','shanghai')]:
        for horizon in (1,4,13):x[stock+'_alpha'+str(horizon)]=x[stock+'_r'+str(horizon)]-x[market+'_r'+str(horizon)]
    for horizon in (1,4,13):x['basket_alpha'+str(horizon)]=(x['zhongtai_alpha'+str(horizon)]+x['tianye_alpha'+str(horizon)])/2
    x['disagreement4']=(x.zhongtai_alpha4-x.tianye_alpha4).abs()
    assert len(x.columns)==42
    frames={}
    for variant in ['Control','Basket','Tianye','Zhongtai']:
        f=x.copy()
        masked=CONTENT if variant=='Control' else [] if variant=='Basket' else [c for c in CONTENT if not c.startswith(variant.lower()+'_')]
        f[masked]=np.nan;f.columns=['equity67_'+c for c in f];frames[variant]=f
    audit['all_valid']=audit[[m+'_valid' for m in MARKETS]].all(axis=1)&audit[[s+'_valid' for s in STOCKS]].any(axis=1)
    return frames,audit
