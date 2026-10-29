"""Matched controls and registered geographic expectation interactions."""
import numpy as np
import pandas as pd
from pvc_intelligence.features.weather_outlook import outlook_at

REGIONS = {'Texas': ['72243', '72255'], 'Louisiana': ['72240', '72232']}
CONTENT = ([f'{lead}_{region}_{kind}' for lead in ['610', '814']
            for region in REGIONS for kind in ['below', 'above', 'winter_below']]
           + [f'{region}_below_lead_difference' for region in REGIONS]
           + [f'{lead}_winter_max_below' for lead in ['610', '814']])


def build_weather_features(rows, calendar, delay_days=0):
    records, audits = [], []
    for date in pd.DatetimeIndex(calendar):
        angle = 2 * np.pi * (date.day_of_year - 15) / 365.25
        winter = max(0., np.cos(angle))
        values = {c: np.nan for c in CONTENT}
        values.update(calendar_sin=np.sin(angle), calendar_cos=np.cos(angle))
        available = outlook_at(rows, date, delay_days)
        issued = {}
        for kind, selected in available.items():
            lead = '610' if kind == 'fcst610' else '814'
            values[lead + '_available'] = int(selected is not None)
            values[lead + '_age_days'] = np.nan
            values[lead + '_valid_start_days'] = np.nan
            values[lead + '_valid_end_days'] = np.nan
            issued[lead] = pd.NaT
            if selected is None:
                continue
            issue = selected.issue_date.iloc[0]
            issued[lead] = issue
            values[lead + '_age_days'] = (date - issue).days
            values[lead + '_valid_start_days'] = (selected.valid_from.iloc[0] - date).days
            values[lead + '_valid_end_days'] = (selected.valid_to.iloc[0] - date).days
            for region, stations in REGIONS.items():
                below = float(selected.loc[stations, 'TBLW'].mean()) / 100
                above = float(selected.loc[stations, 'TABV'].mean()) / 100
                values[f'{lead}_{region}_below'] = below
                values[f'{lead}_{region}_above'] = above
                values[f'{lead}_{region}_winter_below'] = below * winter
            values[lead + '_winter_max_below'] = float(selected.TBLW.max()) / 100 * winter
        for region in REGIONS:
            values[region + '_below_lead_difference'] = (values[f'814_{region}_below'] -
                                                         values[f'610_{region}_below'])
        records.append(values)
        audits.append(dict(week_date=date, issue610=issued['610'], issue814=issued['814'],
                           both_leads_available=all(f is not None for f in available.values())))
    frame = pd.DataFrame(records, index=pd.DatetimeIndex(calendar, name='week_date'))
    frame.columns = ['weather62_' + c for c in frame]
    control = frame.copy()
    control[['weather62_' + c for c in CONTENT]] = np.nan
    assert len(CONTENT) == 16 and len(frame.columns) == 26
    return frame, control, pd.DataFrame(audits).set_index('week_date')
