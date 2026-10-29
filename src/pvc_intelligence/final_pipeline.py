"""Portable, offline raw-workbook → past-only features → fitted 13-week forecast."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from pvc_intelligence.data.pit import asof_snapshot
from pvc_intelligence.features.exogenous import target_features
from pvc_intelligence.features.world_v20 import workbook_markets_explicit_frequency
from pvc_intelligence.features.curve import contract_curve
from pvc_intelligence.features.submission_spot import features as spot_features
from pvc_intelligence.features.producer_equity_v67 import build_features
from pvc_intelligence.models.directional_mixture import state_price_pairs,mature_state_price_data,quantile_model,finite_state_decision


def build(raw_path,public,config):
 cutoff=pd.Timestamp(config['cutoff'])
 if cutoff.dayofweek!=4:raise ValueError('Cutoff must be a Friday')
 raw=pd.read_excel(raw_path,engine='pyxlsb',sheet_name='датасеты')
 raw['report_date']=pd.to_datetime(raw.report_date,unit='D',origin='1899-12-30')
 raw=raw[raw.report_date.le(cutoff)&raw['выборка'].eq('обучающая')&raw.scenario.eq('факт')].copy()
 part=raw[raw.prod_version.eq(config['target'])].sort_values('report_date')
 if part.empty:raise ValueError('Target absent from workbook')
 identity=[c for c in part if c not in ['report_date','k','выборка']]
 if len(part[identity].drop_duplicates())!=1 or part.report_date.duplicated().any():raise ValueError('Ambiguous target identity or duplicate target dates')
 k=part.set_index('report_date').k.resample('W-FRI').last();k=k.reindex(pd.date_range(k.index.min(),cutoff,freq='W-FRI',name='week_date'))
 if pd.isna(k.iloc[-1]):raise ValueError('No observed target at cutoff')
 # Verify offline snapshots before reading. No network or post-cutoff quotes needed.
 for name,meta in json.loads((public/'manifest.json').read_text()).items():
  if hashlib.sha256((public/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Changed public snapshot: '+name)
 s=pd.read_csv(public/'spot.csv');s=s[pd.to_datetime(s.quote_date).lt(cutoff)]
 spot,_=spot_features(s,k)
 daily=pd.read_csv(public/'contracts.csv',parse_dates=['date']);daily=daily[daily.date.lt(cutoff)]
 curve,_=contract_curve(daily,k.index)
 records=pd.read_csv(public/'nbs.csv',parse_dates=['quote_date','available_at']);records=records[records.available_at.le(cutoff)]
 records=records.rename(columns={'price':'value'}).assign(series='nbs_sg5');values,_=asof_snapshot(records,k.index)
 nbs={f'nbs_sg5_r{h}':np.log(values.nbs_sg5/values.nbs_sg5.shift(h)) for h in [1,2,4,8,13]}
 nbs['nbs_sg5_divergence4']=nbs['nbs_sg5_r4']-np.log(k/k.shift(4));nbs['nbs_sg5_missing']=values.nbs_sg5.isna().astype(float)
 selected=[c for c in spot if c.startswith('spot_') or 'basis_' in c]
 base=target_features(k).join(spot[selected]).join(curve[['annualized_curve','far_near_oi_ratio']].add_prefix('curve_')).join(pd.DataFrame(nbs,index=k.index))
 w,_,_=workbook_markets_explicit_frequency(raw,k,delay_weeks=1)
 base=base.join(w[[c for c in w if c.startswith(('wb_c34f7e2674','wb_65ea634a25'))]])
 series={n:pd.read_csv(public/(n+'.csv'),parse_dates=['date']) for n in ['zhongtai','tianye','shanghai','shenzhen']}
 series={n:f[f.date.lt(cutoff)] for n,f in series.items()}
 # Equity source uses its original 2014 calendar for lag/warmup consistency.
 calendar=pd.date_range('2014-01-03',cutoff,freq='W-FRI',name='week_date')
 eq,audit=build_features(series,calendar)
 x=base.join(eq['Basket'].reindex(k.index))[config['columns']]
 if not bool(audit.loc[cutoff,'all_valid']):raise ValueError('Selected external source gate inactive; fail closed, no unvalidated fallback')
 return raw,k,x


def train(raw_path,public,config,out):
 out.mkdir(parents=True,exist_ok=True);raw,k,x=build(raw_path,public,config);cutoff=pd.Timestamp(config['cutoff'])
 X,y,meta=mature_state_price_data(x,state_price_pairs(k),cutoff)
 if not meta.label_date.lt(cutoff).all():raise AssertionError('Future training label')
 model=quantile_model(config['parameters']).fit(X,y);model.save_model(str(out/'model.cbm'))
 receipt=dict(config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),model=config['model'],cutoff=config['cutoff'],target=config['target'],raw_rows=len(raw),observed_weeks=int(k.notna().sum()),grid_weeks=len(k),training_pairs=len(meta),last_label=str(meta.label_date.max().date()),feature_count=len(x.columns),last_price=float(k.iloc[-1]),private_raw_sha256=hashlib.sha256(Path(raw_path).read_bytes()).hexdigest())
 # Private derived history stays in local build, never in publication package.
 x.loc[[cutoff]].to_csv(out/'query_private.csv');k.to_csv(out/'target_private.csv')
 (out/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 return model,k,x


def predict(model,k,x,config,out):
 cutoff=pd.Timestamp(config['cutoff']);q=pd.concat([x.loc[[cutoff]]]*13,ignore_index=True).assign(horizon=range(1,14))
 uq=np.asarray(model.predict(q));log,_=finite_state_decision(np.ones((13,1)),uq[:,None,:]);pred=float(k.loc[cutoff])*np.exp(log)
 f=pd.DataFrame(dict(week_start=pd.date_range(cutoff+pd.Timedelta(weeks=1),periods=13,freq='W-FRI'),horizon=range(1,14),forecast=pred))
 out.mkdir(parents=True,exist_ok=True);f.to_csv(out/'forecast.csv',sep=';',encoding='utf-8-sig',index=False,float_format='%.15g')
 # No calibrated intervals are claimed from quantile support alone.
 pd.DataFrame(dict(week_start=f.week_start,position=np.zeros(13,dtype=int))).to_csv(out/'strategy.csv',sep=';',encoding='utf-8-sig',index=False)
 return f


def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['run','train','predict']);p.add_argument('--raw',type=Path,default=Path('data/pvc_raw.xlsb'));p.add_argument('--config',type=Path,default=Path('submission/config.json'));p.add_argument('--public',type=Path,default=Path('submission/public_inputs'));p.add_argument('--model-dir',type=Path,default=Path('build/final'));p.add_argument('--out',type=Path,default=Path('results'));a=p.parse_args();c=json.loads(a.config.read_text())
 if a.command in ['run','train']:model,k,x=train(a.raw,a.public,c,a.model_dir)
 else:
  receipt=json.loads((a.model_dir/'training_receipt.json').read_text())
  if receipt.get('config_sha256')!=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest():raise ValueError('Model/config mismatch; retrain first')
  if receipt['private_raw_sha256']!=hashlib.sha256(a.raw.read_bytes()).hexdigest():raise ValueError('Workbook changed; retrain first')
  _,k,x=build(a.raw,a.public,c);model=CatBoostRegressor();model.load_model(str(a.model_dir/'model.cbm'))
 if a.command in ['run','predict']:print(predict(model,k,x,c,a.out).to_string(index=False))
if __name__=='__main__':main()
