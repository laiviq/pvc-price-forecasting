from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-07-03")
TARGET = "ПВХ SG-5 (Китай) DAP Достык, USD/т"
SERIES = {
    "naphtha_cif": "Нафта CIF NWE, $/тн",
    "naphtha_fob": "Нафта FOB Rtd, $/тн",
    "brent": "Ice Brent month 5 day rolling avg",
    "ethylene_spot": "Ethylene (spot) Ex-Dahej",
    "ethylene_contract": "Ethylene (Contract) Ex-Dahej",
    "pvc_shanghai": "PVC Pipe Grade CFR Shanghai",
    "pvc_sg3_dostyk": "ПВХ SG-3 (Китай) DAP Достык, USD/т",
    "pvc_nanjing_semi": "PVC Semi Rigid Grade K67 Ex-Nanjing",
    "pvc_nanjing_flexible": "PVC Suspension Flexible Grade K67 Ex-Nanjing",
    "ethylene_shanghai": "Ethylene CFR Shanghai",
    "pvc_busan": "PVC Pipe Grade FOB Busan",
}
WEEKLY = {"ethylene_spot", "ethylene_contract", "pvc_shanghai",
          "pvc_nanjing_semi", "pvc_nanjing_flexible", "ethylene_shanghai", "pvc_busan"}


def read_raw(root: Path) -> pd.DataFrame:
    raw = pd.read_excel(root / "data/pvc_raw.xlsb", engine="pyxlsb", sheet_name="датасеты")
    raw["report_date"] = pd.to_datetime(raw.report_date, unit="D", origin="1899-12-30")
    # Filter before any resampling or feature construction. Never ingest hidden outcomes.
    return raw.loc[(raw.report_date <= CUTOFF) & raw["выборка"].eq("обучающая")
                   & raw.scenario.eq("факт")].copy()


def asof_snapshot(records, cutoffs, *, verified_only=True, max_age_days=21):
    """Long records -> values and provenance. No backfill; revisions use their release date.

    `available_at` is an information date, not a quote date. Unverified records
    are excluded by default. Research proxies require an explicit opt-in.
    """
    r = records.copy()
    for c in ["quote_date", "available_at"]:
        r[c] = pd.to_datetime(r[c])
    if r.available_at.isna().any() or (r.available_at < r.quote_date).any():
        raise ValueError("Missing release dates or release before observation")
    if r.duplicated(["series", "quote_date", "available_at"]).any():
        raise ValueError("Ambiguous quote/release version")
    if verified_only:
        r = r.loc[r.availability_verified.eq(True)]
    dates = pd.DatetimeIndex(cutoffs, name="week_date")
    out, lineage = pd.DataFrame(index=dates), []
    for name, part in r.groupby("series", sort=True):
        # At each cutoff, select latest observation, then latest available revision.
        rows = []
        for cutoff in dates:
            known = part.loc[(part.available_at <= cutoff) & (part.quote_date <= cutoff)]
            if known.empty:
                rows.append(np.nan)
                continue
            row = known.sort_values(["quote_date", "available_at"]).iloc[-1]
            age = (cutoff - row.quote_date).days
            valid = age <= max_age_days
            rows.append(row.value if valid else np.nan)
            lineage.append({"week_date": cutoff, "series": name, "quote_date": row.quote_date,
                            "available_at": row.available_at, "age_days": age,
                            "usable": valid, "availability_verified": row.availability_verified,
                            "source_id": row.source_id})
        out[name] = rows
    return out, pd.DataFrame(lineage)


def build_panel(root: Path, market_delay_weeks=1):
    """Known target calendar + explicitly unverified lagged market proxies.

    Publication timestamps for the original workbook were not supplied. These
    are sensitivity-study inputs, not historically verified real-time vintages.
    """
    t = pd.read_csv(root / "data/processed/target_weekly.csv", parse_dates=["week_date"])
    t = t.loc[t.week_date <= CUTOFF].set_index("week_date").sort_index()
    if not t.index.is_unique or not t.index.equals(pd.date_range(t.index.min(), CUTOFF, freq="W-FRI", name="week_date")):
        raise ValueError("Expected unique continuous Friday grid")
    panel = t[["k"]].copy()
    raw = read_raw(root)
    all_records = []
    meta = ["prod_version", "frequency", "basis", "currency", "measure_unit", "location", "country", "port_name_ru"]
    for name, product in SERIES.items():
        part = raw.loc[raw.prod_version.eq(product)].copy()
        if name in WEEKLY:
            part = part.loc[part.frequency.eq("еженедельно")]
        if part[meta].drop_duplicates().shape[0] != 1 or part.report_date.duplicated().any():
            raise ValueError(f"Mixed identity or duplicate dates: {name}")
        # Match old one-week lag, but retain quote and assumed availability dates.
        week = part.report_date + pd.to_timedelta((4 - part.report_date.dt.dayofweek) % 7, unit="D")
        for i, row in part.iterrows():
            all_records.append({"series": name, "quote_date": row.report_date,
                "available_at": week.loc[i] + pd.Timedelta(weeks=market_delay_weeks),
                "value": row.k, "availability_verified": False, "source_id": "original_xlsb",
                "unit": "k_unknown_transform", "basis": row.basis})
    daily = pd.read_csv(root / "data/external/pvc_futures_daily.csv", parse_dates=["date"])
    daily = daily.loc[daily.date <= CUTOFF - pd.Timedelta(days=1)]
    for row in daily.itertuples():
        all_records.append({"series": "futures", "quote_date": row.date,
            "available_at": row.date + pd.Timedelta(days=1), "value": row.close,
            "availability_verified": False, "source_id": "sina_v0_akshare",
            "unit": "CNY/t", "basis": "DCE_continuous_unadjusted_unknown_roll"})
    fx = pd.read_csv(root / "data/external/fx_weekly.csv", parse_dates=["fx_date", "week_date"])
    fx = fx.loc[fx.week_date <= CUTOFF].dropna(subset=["fx_date", "cny_per_usd"]).drop_duplicates("fx_date")
    for row in fx.itertuples():
        all_records.append({"series": "fx", "quote_date": row.fx_date,
            "available_at": row.fx_date + pd.Timedelta(days=1), "value": row.cny_per_usd,
            "availability_verified": False, "source_id": "ecb_cross_retrieved_vintage",
            "unit": "CNY/USD", "basis": "reference_fx"})
    records = pd.DataFrame(all_records)
    values, lineage = asof_snapshot(records, panel.index, verified_only=False)
    panel = panel.join(values, validate="one_to_one")
    # Target is NEVER imputed. Feature as-of carry is capped at 21 calendar days.
    return panel, records, lineage
