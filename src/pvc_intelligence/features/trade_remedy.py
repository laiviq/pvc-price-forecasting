"""Published legal milestones; recommendations never manufacture charged duties."""
import pandas as pd

STAGES = {'initiation', 'preliminary_recommendation', 'final_recommendation',
          'termination', 'investigation_deadline_extension'}


def first_friday_after(date):
    date = pd.Timestamp(date).normalize()
    days = (4 - date.weekday()) % 7
    return date + pd.Timedelta(days=days or 7)


def information_date(printed_issue, catalog_publication, metadata_dates=(), delay_weeks=0):
    """Use explicit publication/copy bounds, never an earlier date cited in prose."""
    dates = []
    for value in [printed_issue, catalog_publication, *metadata_dates]:
        if value is None or pd.isna(value):
            continue
        date = pd.Timestamp(value)
        if date.year < 2000 or date.year > 2026:
            raise ValueError('Out-of-scope date; no automatic year repair')
        if date.tzinfo is not None:
            date = date.tz_convert('Asia/Kolkata').tz_localize(None)
        dates.append(date.normalize())
    if printed_issue is None or not dates:
        raise ValueError('A verified printed issue date is required')
    return first_friday_after(max(dates)) + pd.Timedelta(weeks=delay_weeks)


def validate_milestone(fact):
    if fact['stage'] not in STAGES:
        raise ValueError('Unsupported legal milestone')
    if fact['product_class'] != 'ordinary_suspension_scope' or not fact['scope_review_pass']:
        raise ValueError('Grade applicability has not passed review')
    # Legal metadata titles can point to an entirely different chemical.
    if not fact.get('body_identity_review_pass', False):
        raise ValueError('Body/case identity mismatch or unreviewed source')
    if fact.get('this_document_imposes_duty', False):
        raise ValueError('DGTR milestone does not prove an implemented duty')
    if fact['stage'] == 'investigation_deadline_extension' and not fact.get('known_deadline'):
        raise ValueError('No explicit investigation completion deadline')


def tracked_cases_at(facts, cutoff):
    """State of covered proceedings only; not an assertion about all market duties."""
    cutoff = pd.Timestamp(cutoff)
    known = []
    for fact in facts:
        validate_milestone(fact)
        if pd.Timestamp(fact['available_date']) <= cutoff:
            known.append(fact)
    states = {}
    for fact in sorted(known, key=lambda f: (f['available_date'], f['event_id'])):
        key = fact['case_id']
        state = states.setdefault(key, dict(pending=False, preliminary_seen=False,
                                            known_deadline=None, implemented_duty_status='unknown_in_this_ledger'))
        if fact['stage'] == 'initiation':
            state['pending'] = True
        elif fact['stage'] == 'preliminary_recommendation':
            state['preliminary_seen'] = True
        elif fact['stage'] in ['final_recommendation', 'termination']:
            state['pending'] = False
        elif fact['stage'] == 'investigation_deadline_extension':
            state['known_deadline'] = fact['known_deadline']
        state['last_publication'] = fact['available_date']
    return states
