"""Auditable, conservative extraction from dated PVC weekly publications."""
import re

import numpy as np
import pandas as pd

NUMBER = r"([+-]?\d+(?:\.\d+)?)"

# Only quoted levels, never arbitrary numbers in tables or plotted chart axes.
PATTERNS = {
    "utilization_pct": ("supply_demand", "percent", [
        r"PVC(?:粉)?(?:行业)?(?:整体)?(?:产能利用率|开工负荷率|开工负荷)(?:为|在)?" + NUMBER + "%",
        r"PVC(?:粉)?(?:行业)?(?:整体)?(?:产能利用率|开工负荷率)[^。；]{0,60}?(?:至|为|在)" + NUMBER + "%",
        r"供应端[,：][^。；]{0,150}?(?<!法)产能利用率[^。；]{0,40}?(?:至|为|在)" + NUMBER + "%"]),
    "weekly_output_10kt": ("supply_demand", "10000_tonnes", [
        r"本周(?:国内)?(?:PVC(?:粉)?)?产量(?:为|在)?" + NUMBER + "万吨"]),
    "inventory_original_10kt": ("supply_demand", "10000_tonnes_original_sample", [
        r"华东及华南(?:原样本|样本)(?:仓库)?总库存(?:为|在)?" + NUMBER + "万吨",
        r"PVC社会库存(?:（(?:隆众)?21家样本）|\((?:隆众)?21家样本\))[^。；]{0,50}?(?:至|在|为)" + NUMBER + "万吨"]),
    "inventory_expanded_10kt": ("supply_demand", "10000_tonnes_expanded_sample", [
        r"华东及华南扩充后样本(?:仓库)?总库存(?:为|在)?" + NUMBER + "万吨"]),
    "carbide_cost_cny_t": ("cost", "CNY/tonne", [
        r"电石法(?:全国)?(?:平均)?成本[^。；]{0,50}?(?:至|在|为)" + NUMBER + "元/吨"]),
    "ethylene_cost_cny_t": ("cost", "CNY/tonne", [
        r"乙烯法(?:全国)?(?:平均)?成本[^。；]{0,50}?(?:至|在|为)" + NUMBER + "元/吨"]),
    "wuhai_carbide_cny_t": ("cost", "CNY/tonne", [
        r"乌海(?:地区)?(?:给贸易环节|给贸易商|贸易|地区贸易)?电石出厂价(?:格)?(?:在|为)?" + NUMBER + "元/吨",
        r"乌海[^。；]{0,30}?电石出厂价(?:格)?(?:在|为)?" + NUMBER + "元/吨"]),
    "caustic_soda_32_cny_t": ("cost", "CNY/tonne_32percent_liquid_Shandong", [
        r"山东32%离子膜碱周均价(?:为|在)?" + NUMBER + "元/吨"]),
    "fob_tianjin_usd_t": ("trade", "USD/tonne_FOB_Tianjin_carbide_SG5", [
        r"电石法(?:5型)?FOB天津出口(?:成交)?价格评估(?:在|为)?" + NUMBER + "美元/吨",
        r"电石法FOB出口成交价格评估(?:在|为)?" + NUMBER + "美元/吨"]),
    "shandong_profit_cny_t": ("trade", "CNY/tonne_Shandong_carbide_PVC_profit", [
        r"山东地区PVC(?:粉)?盈利(?:在|为)?" + NUMBER + "元/吨"]),
    "inner_mongolia_profit_cny_t": ("trade", "CNY/tonne_Inner_Mongolia_carbide_PVC_profit", [
        r"内蒙(?:古)?地区PVC(?:粉)?盈利(?:在|为)?" + NUMBER + r"(?:元)?/吨"]),
    "basis_cny_t": ("trade", "CNY/tonne_spot_minus_front_futures", [
        r"主力基差(?:主力基差)?(?:为|在)?" + NUMBER + r"[,。；元]",
        r"主力基差主力基(?:为|在)?" + NUMBER + r"[,。；元]"]),
}

LEXICON = {
    "logistics": ["运输", "车皮", "运费", "物流", "到货", "红海", "霍尔木兹"],
    "supply_disruption": ["限电", "停车", "检修", "降负荷"],
    "supply_restart": ["重启", "复产", "投产", "恢复开工"],
    "export_policy": ["印度", "反倾销", "BIS", "关税"],
    "forward_modality": ["下周", "预计", "计划", "预期"],
    "demand_weak": ["订单不佳", "需求疲软", "需求偏弱", "需求低迷", "需求走弱"],
    "demand_improving": ["订单增加", "需求回暖", "需求改善", "接单较好", "接单放量"],
}


def extract_numeric(text):
    """Return value + exact matched evidence; no future analyst forecasts as facts."""
    # Numeric observations only before the forward-looking section.
    factual = re.split(r"展望[：:]|观点[：:]?|整体来看[：:]", text)[0]
    rows = []
    for name, (group, unit, patterns) in PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, factual)
            if match:
                value = float(match.group(1))
                if (name.endswith("_pct") and not 0 <= value <= 100) or (
                    "profit" not in name and "basis" not in name and value <= 0):
                    break
                rows.append({"variable": name, "group": group, "unit": unit, "value": value,
                             "evidence": match.group(0), "pattern": pattern})
                break
    return rows


def extract_text(text):
    """Transparent corpus-specific counts; not an all-news sentiment score."""
    parts = re.split(r"展望[：:]|观点[：:]?", text, maxsplit=1)
    outlook = parts[-1] if len(parts) > 1 else ""
    outlook = re.split(r"研究员|重要事项|免责声明", outlook)[0]
    if len(outlook) < 30:
        return {}
    return {name: 1000 * sum(outlook.count(word) for word in words) / len(outlook)
            for name, words in LEXICON.items()}


def released_features(records, calendar, *, delay_weeks=0, max_age_days=21):
    """One observation per source/variable/publication; strictly past releases."""
    dates = pd.DatetimeIndex(calendar, name="week_date")
    columns = {}
    lineage = []
    r = records.copy()
    r["published_date"] = pd.to_datetime(r.published_date)
    days = (4 - r.published_date.dt.day_of_week) % 7
    r["available_at"] = r.published_date + pd.to_timedelta(days.where(days.ne(0), 7), unit="D") + pd.Timedelta(weeks=delay_weeks)
    if r.duplicated(["series", "available_at"]).any():
        raise ValueError("Ambiguous publication versions for a source series")
    for name, part in r.groupby("series", sort=True):
        part = part.sort_values("available_at")
        values, age_values = [], []
        for cutoff in dates:
            known = part.loc[part.available_at <= cutoff]
            if known.empty:
                values.append(np.nan); age_values.append(np.nan)
                continue
            row = known.iloc[-1]
            age = (cutoff - row.available_at).days
            usable = age <= max_age_days
            values.append(row.value if usable else np.nan)
            age_values.append(age if usable else np.nan)
            if usable:
                lineage.append({"week_date": cutoff, "series": name, "available_at": row.available_at,
                    "published_date": row.published_date, "report_id": row.report_id, "source_url": row.source_url})
        series = pd.Series(values, index=dates, dtype=float)
        columns[name] = series
        columns[name + "_d4"] = series - series.shift(4)
        columns[name + "_age"] = age_values
        columns[name + "_missing"] = series.isna().astype(float)
    return pd.DataFrame(columns, index=dates), pd.DataFrame(lineage)
