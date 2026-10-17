"""Replace the crop price and production cost of finished runs by the NRW planning data, without new physics.

Source: Landwirtschaftskammer Nordrhein-Westfalen (2024), Planungsdeckungsbeiträge 2024/2025.
Price = mean producer price 2019-2024; cost = variable costs (direct costs + variable machinery costs) at the yield
level closest to the simulated open-field yield. Soybean is not covered (conventional) and keeps its assumption.

    python analysis/update_crop_economics.py            # backs up each run to <run>_econ_assumption first
Recomputes crop revenue, crop cost, revenue, cash flow, net income and the Pareto flags of every listed run.
"""
import json
import shutil
import sys

import pandas as pd

from common import RESULTS, crf, mark_pareto

NRW = {  # crop key in folder suffix: (price EUR/t, cost EUR/ha)
    "": (213.1, 1120.),                         # winter wheat: 21.31 EUR/dt; variable costs at 90 dt/ha
    "_barley_Spring_barley_301": (196.3, 890.),  # 19.63 EUR/dt; 70 dt/ha (highest level)
    "_maize_Grain_maize_201": (218.7, 1690.),    # 21.87 EUR/dt; 120 dt/ha (highest level)
    "_potato_Potato_701": (197.9, 4930.),        # processing potatoes 19.79 EUR/dt; 600 dt/ha (highest level)
    "_sugarbeet_Sugarbeet_601": (42.0, 1620.),   # 4.20 EUR/dt; 850 dt/ha
}
BASES = ["recheck_combined_192skies", "single_axis_recheck_combined_192skies", "vertical_recheck_combined_192skies"]


def update(folder, price, cost):
    d = RESULTS/folder
    backup = RESULTS/(folder + "_econ_assumption")
    if not backup.exists():
        shutil.copytree(d, backup)
    meta = json.loads((backup/"run_metadata.json").read_text(encoding="utf-8"))
    c = meta["config"]
    old = (c["crop_price_eur_t"], c["crop_cost_eur_planted_ha"])
    c["crop_price_eur_t"], c["crop_cost_eur_planted_ha"] = price, cost
    meta.setdefault("notes", []).append("crop price/cost replaced by NRW planning data (update_crop_economics.py)")
    for name in ("evaluated_candidates.csv", "pareto_front.csv"):
        f = backup/name
        if not f.exists():
            continue
        t = pd.read_csv(f, float_precision="round_trip")
        planted = t.planted_area_ha
        t["crop_revenue_eur"] = t.crop_yield_t_ha.fillna(0.)*planted*price
        t["crop_cost_eur"] = planted*cost
        t["revenue_eur"] = t.electrical_revenue_eur + t.agripv_bonus_eur + t.crop_revenue_eur
        t["annual_operating_cashflow_eur"] = t.revenue_eur - t.annual_opex_eur - t.crop_cost_eur
        t["annual_net_eur"] = t.annual_operating_cashflow_eur - t.capex_total_eur*crf(c["discount_rate"], c["lifetime_years"])
        t["annual_net_eur_ha"] = t.annual_net_eur/t.footprint_ha
        t["payback_period_years"] = (t.capex_total_eur/t.annual_operating_cashflow_eur).where(t.annual_operating_cashflow_eur > 0)
        if name == "evaluated_candidates.csv":
            t = mark_pareto(t.drop(columns="pareto", errors="ignore"))
            t.to_csv(d/name, index=False)
            t[t.pareto.astype(bool)].to_csv(d/"pareto_front.csv", index=False)
    (d/"run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{folder}: price {old[0]} -> {price}, cost {old[1]} -> {cost}")


if __name__ == "__main__":
    for suffix, (price, cost) in NRW.items():
        for base in BASES:
            if (RESULTS/(base + suffix)).exists():
                update(base + suffix, price, cost)
            else:
                print("missing", base + suffix, file=sys.stderr)
