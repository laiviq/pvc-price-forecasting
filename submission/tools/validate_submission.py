from pathlib import Path
import json,sys
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from pvc_intelligence.evaluation.submission import validate_forecast,validate_strategy
f=pd.read_csv(R/'results/forecast.csv',sep=';',parse_dates=['week_start']);s=pd.read_csv(R/'results/strategy.csv',sep=';',parse_dates=['week_start'])
assert list(f)==['week_start','horizon','forecast'];assert list(s)==['week_start','position'];assert len(f)==len(s)==13
validate_forecast(f,pd.Timestamp('2026-07-03'));validate_strategy(s,pd.Timestamp('2026-07-03'))
assert f.week_start.equals(s.week_start);assert s.position.eq(0).all();assert np.isfinite(f.forecast).all()
for p in ['forecast.csv','strategy.csv']:assert (R/'results'/p).read_bytes().startswith(b'\xef\xbb\xbf')
receipt=dict(status='pass',forecast_rows=13,strategy_rows=13,first_week=str(f.week_start.min().date()),last_week=str(f.week_start.max().date()),strict_columns=True,encoding='UTF-8 BOM',separator=';',finite_positive_forecast=True,positions=[0]*13)
(R/'submission/evidence/submission_validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
