"""Same-contract momentum and simultaneous price-curve features."""
import numpy as np
import pandas as pd


def contract_curve(records, calendar, *, delay_weeks=0):
    r = records.copy()
    r["date"] = pd.to_datetime(r.date)
    if r.duplicated(["contract", "date"]).any():
        raise ValueError("Duplicate individual contract trading dates")
    histories = {name: p.set_index("date").sort_index() for name,p in r.groupby("contract")}
    rows, lineage = [], []
    for cutoff in pd.DatetimeIndex(calendar):
        asof = cutoff-pd.Timedelta(days=1, weeks=delay_weeks)
        candidates = []
        for name, history in histories.items():
            maturity = pd.Timestamp(2000+int(name[1:3]), int(name[3:5]), 1)
            days = (maturity-asof).days
            if not 30 <= days <= 370:
                continue
            known = history.loc[:asof]
            if known.empty:
                continue
            quote = known.iloc[-1]
            age = (asof-known.index[-1]).days
            if age > 7 or quote.close <= 0 or quote.volume <= 0 or quote.open_interest <= 0:
                continue
            candidates.append((maturity, name, known, quote, age))
        candidates.sort(key=lambda p:p[0])
        row = {"week_date": cutoff, "nodes_available": float(len(candidates)), "market_missing": float(not candidates)}
        if candidates:
            maturity, name, known, quote, age = candidates[0]
            row.update(quote_age=float(age), log_volume=float(np.log1p(quote.volume)), log_open_interest=float(np.log1p(quote.open_interest)))
            for weeks in [1,4,13]:
                previous = known.loc[:asof-pd.Timedelta(weeks=weeks)]
                if len(previous) and (asof-pd.Timedelta(weeks=weeks)-previous.index[-1]).days <= 7:
                    row[f"same_contract_return{weeks}"] = float(np.log(quote.close/previous.close.iloc[-1]))
            last = known.tail(21)
            daily = np.log(last.close/last.close.shift(1))
            row["volatility20"] = float(daily.std()) if daily.count() >= 10 else np.nan
            row["intraday_range"] = float((quote.high-quote.low)/quote.close)
            lineage.append({"week_date": cutoff, "role": "near", "contract": name, "quote_date": known.index[-1], "asof": asof,
                "maturity_anchor": maturity, "close": quote.close})
            if len(candidates)>1:
                far_maturity, far_name, far_known, far, _ = candidates[1]
                # Same date is required for a curve spread. Never compare stale
                # near and fresh far quotes as if simultaneously tradable.
                if far_known.index[-1] == known.index[-1]:
                    row["annualized_curve"] = float(np.log(far.close/quote.close)*365/(far_maturity-maturity).days)
                    row["far_near_oi_ratio"] = float(np.log(far.open_interest/quote.open_interest))
                    lineage.append({"week_date": cutoff, "role": "far", "contract": far_name, "quote_date": far_known.index[-1], "asof": asof,
                        "maturity_anchor": far_maturity, "close": far.close})
        rows.append(row)
    return pd.DataFrame(rows).set_index("week_date"), pd.DataFrame(lineage)
