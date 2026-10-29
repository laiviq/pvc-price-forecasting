"""Operational fallback in units of k; no claim of commercial validation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


def export_baseline(root: Path):
    source = root / "data/processed/target_weekly.csv"
    t = pd.read_csv(source, parse_dates=["week_date"])
    cutoff = pd.Timestamp("2026-07-03")
    known = t.loc[t.week_date <= cutoff].sort_values("week_date")
    current = known.loc[known.week_date.eq(cutoff), "k"]
    if len(current) != 1 or current.isna().any() or current.iloc[0] <= 0:
        raise ValueError("No unique observed positive target at information cutoff")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = [{"model": "Naive_v2_fallback", "information_cutoff": cutoff,
             "horizon": h, "forecast_date": cutoff + pd.Timedelta(weeks=h),
             "prediction_k": current.iloc[0], "unit": "k_unknown_price_transform",
             "commercial_readiness": False, "data_sha256": digest} for h in range(1, 14)]
    out = root / "results/forecast_baseline_v2.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (root / "results/research_v2/baseline_model.json").write_text(json.dumps({
        "model": "Naive_v2_fallback", "information_cutoff": str(cutoff.date()),
        "last_observed_k": float(current.iloc[0]), "data_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "No candidate passed broad-history and temporal promotion gates; target transform remains unknown",
        "commercial_readiness": False}, indent=2))
    return out
