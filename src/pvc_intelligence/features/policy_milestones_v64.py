"""Semantic stage features from published tracked cases; no implemented tariff inference."""
import numpy as np
import pandas as pd
from pvc_intelligence.features.trade_remedy import validate_milestone

CONTENT = ['china_pending', 'other_pending', 'china_preliminary_pending', 'china_oldest_age13',
           'china_initiations4', 'china_initiations13', 'other_initiations13',
           'china_preliminary13', 'china_final13', 'other_termination13',
           'china_deadline_within13', 'china_next_deadline_fraction']


def at_date(facts, cutoff, delay_days=0):
    cutoff = pd.Timestamp(cutoff).normalize()
    known = []
    for row in facts:
        validate_milestone(row)
        publication = pd.Timestamp(row['available_date']) + pd.Timedelta(days=delay_days)
        if publication <= cutoff:
            known.append(dict(row, publication=publication))
    known.sort(key=lambda r: (r['publication'], r['event_id']))
    values = {name: 0. for name in CONTENT}
    values['china_next_deadline_fraction'] = np.nan
    angle = 2 * np.pi * (cutoff.day_of_year - 1) / 365.25
    control = dict(publications_log=np.log1p(len(known)), publications4=0., publications13=0.,
                   publication_age52=np.nan, calendar_sin=np.sin(angle), calendar_cos=np.cos(angle))
    states = {}
    for row in known:
        age = (cutoff - row['publication']).days / 7
        if age <= 26:
            control['publications4'] += np.exp(-age / 4)
            control['publications13'] += np.exp(-age / 13)
            names = []
            if row['stage'] == 'initiation':
                if row['china_in_scope']: names += [('china_initiations4', 4), ('china_initiations13', 13)]
                if row['other_origin_case']: names += [('other_initiations13', 13)]
            if row['stage'] == 'preliminary_recommendation' and row['china_in_scope']:
                names += [('china_preliminary13', 13)]
            if row['stage'] == 'final_recommendation' and row['china_in_scope']:
                names += [('china_final13', 13)]
            if row['stage'] == 'termination' and row['other_origin_case']:
                names += [('other_termination13', 13)]
            for name, decay in names: values[name] += np.exp(-age / decay)
        state = states.setdefault(row['case_id'], dict(pending=False, preliminary=False, start=None,
                deadline=None, china=row['china_in_scope'], other=row['other_origin_case']))
        if row['stage'] == 'initiation':
            state.update(pending=True, preliminary=False, start=row['publication'], deadline=None)
        elif row['stage'] == 'preliminary_recommendation': state['preliminary'] = True
        elif row['stage'] in ['final_recommendation', 'termination']: state['pending'] = False
        elif row['stage'] == 'investigation_deadline_extension': state['deadline'] = pd.Timestamp(row['known_deadline'])
    if known: control['publication_age52'] = min((cutoff-known[-1]['publication']).days/7, 52.)
    deadlines = []
    for state in states.values():
        if not state['pending']: continue
        if state['other']: values['other_pending'] += 1
        if state['china']:
            values['china_pending'] += 1
            values['china_preliminary_pending'] += int(state['preliminary'])
            if state['start'] is not None:
                values['china_oldest_age13'] = max(values['china_oldest_age13'], min((cutoff-state['start']).days/91,4.))
            if state['deadline'] is not None:
                days = (state['deadline']-cutoff).days
                if 0 <= days <= 91: deadlines.append(days)
    values['china_deadline_within13'] = float(len(deadlines))
    if deadlines: values['china_next_deadline_fraction'] = min(deadlines)/91
    audit = dict(known_milestones=len(known), last_available=known[-1]['publication'] if known else pd.NaT,
                 known_case_groups=len(states), implemented_duty_status='not_inferred')
    return {**control, **values}, audit


def build_features(facts, calendar, delay_days=0):
    records, audit = [], []
    for date in pd.DatetimeIndex(calendar):
        values, meta = at_date(facts, date, delay_days)
        records.append(values); audit.append(dict(week_date=date, **meta))
    frame = pd.DataFrame(records, index=pd.DatetimeIndex(calendar, name='week_date'))
    frame.columns = ['policy64_' + name for name in frame]
    control = frame.copy()
    control[['policy64_' + name for name in CONTENT]] = np.nan
    assert len(frame.columns) == 18
    return frame, control, pd.DataFrame(audit).set_index('week_date')
