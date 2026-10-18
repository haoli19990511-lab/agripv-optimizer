"""Electricity price years and curtailment at negative prices, from the cached hourly yield (no physics).
    python analysis/price_scenarios.py recheck_combined_192skies
Revenue is linear in the price, so feeding in at all times with the multi-year mean profile equals the mean of
the single-year revenues. Curtailment (no feed-in and no agri-PV bonus while the price is negative) is not
linear, so it is evaluated year by year and then averaged.
Writes results/<run>/analysis/price_scenarios.csv (all designs) and price_scenarios_summary.csv.
"""
import json
import sys

import numpy as np
import pandas as pd

from common import ROOT, check_reproduction, load_run, mark_pareto

sys.path.insert(0, str(ROOT))
from weighted_engine import load_price_profile  # noqa: E402

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SCENARIOS = {  # name: years averaged
    "2023-2025 (base)": [2023, 2024, 2025], "2020-2022": [2020, 2021, 2022],
    "2023": [2023], "2024": [2024], "2025": [2025],
}
DESIGN_KEYS = ("tilt", "pitch", "azimuth", "clearance_height")


def cached_hourly(row, manifests, base_price):
    """Hourly result of this design; older cache entries (other model versions) must not reproduce the revenue."""
    hits = [f for f, d in manifests if all(abs(d[k]-row[k]) < 1e-9 for k in DESIGN_KEYS)]
    for f in hits:
        hourly = pd.read_csv(f/"hourly.csv.gz")
        inc = sum(hourly[f"{row.pv_type}_array_{s}_effective_incident_energy_Wh_hour"].to_numpy(float)
                  for s in ("front", "rear"))
        kwh = inc*row.electrical_kwh/inc.sum()
        if abs(float(kwh @ base_price(hourly))-row.electrical_revenue_eur) <= 1e-6*abs(row.electrical_revenue_eur):
            return hourly, kwh
    raise FileNotFoundError(f"No cached hourly yield reproduces {row.design_id} ({len(hits)} candidates)")


def main(run):
    table, meta, folder = load_run(run)
    cfg = meta["config"]
    check_reproduction(table, cfg)
    s = meta["sampling"]
    manifests = []
    for m in (ROOT/"results"/"cache_v5").glob("*/manifest.json"):
        j = json.loads(m.read_text())
        if (j.get("sampling", {}).get("days_per_month"), j["sampling"].get("hours_per_day")) == \
                (s["days_per_month"], s["hours_per_day"]) and "tilt" in j.get("design", {}):
            manifests.append((m.parent, j["design"]))
    prices, rows, cache = {}, [], {}

    def base_price(hourly):
        if "base" not in cache:
            hours = pd.DatetimeIndex(pd.to_datetime(hourly.hour_start))
            cache["base"] = load_price_profile(hours, ROOT/"prices", "csv", None,
                                               meta["price_assumptions"]["years"]).to_numpy()
            prices.update({y: load_price_profile(hours, ROOT/"prices", "csv", None, [y]).to_numpy() for y in YEARS})
        return cache["base"]

    for _, r in table.iterrows():
        hourly, kwh = cached_hourly(r, manifests, base_price)  # kWh with the engine's conversion factors
        rate = r.agripv_bonus_eur/r.electrical_kwh if r.electrical_kwh else 0.
        other = r.annual_net_eur-r.electrical_revenue_eur-r.agripv_bonus_eur  # crop, O&M, CAPEX: unchanged
        for name, years in SCENARIOS.items():
            for mode in ("feed-in always", "curtailed at p<0"):
                rev, bonus = [], []
                for y in years:
                    p = prices[y]
                    paid = p >= 0 if mode.startswith("curtailed") else np.ones_like(p, bool)
                    rev.append(float(kwh[paid] @ p[paid]))
                    bonus.append(rate*float(kwh[paid].sum()))
                net = other+np.mean(rev)+np.mean(bonus)
                rows.append(dict(scenario=name, operation=mode, design_id=r.design_id, pv_type=r.pv_type,
                                 n_rows=r.n_rows, LER=r.LER, electrical_revenue_eur=np.mean(rev),
                                 capture_price_ct=100*np.mean(rev)/r.electrical_kwh,
                                 annual_net_eur=net, annual_net_eur_ha=net/r.footprint_ha, feasible=r.feasible))
    out = pd.DataFrame(rows)
    base_net = out[(out.scenario == "2023-2025 (base)") & (out.operation == "feed-in always")].set_index(
        ["design_id", "pv_type"]).annual_net_eur_ha
    ref = table.set_index(["design_id", "pv_type"]).annual_net_eur_ha
    if np.max(np.abs(base_net-ref.reindex(base_net.index))) > 1e-6*ref.abs().max():
        raise AssertionError("Base scenario does not reproduce the engine")
    base_pareto = mark_pareto(table)
    base_set = set(zip(base_pareto[base_pareto.pareto].design_id, base_pareto[base_pareto.pareto].pv_type))
    summary = []
    parts = []
    for (name, mode), g in out.groupby(["scenario", "operation"], sort=False):
        g = mark_pareto(g.reset_index(drop=True))
        parts.append(g)
        cur = set(zip(g[g.pareto].design_id, g[g.pareto].pv_type))
        conv = g[g.pv_type == "conventional"]
        best_inc = conv.loc[conv.annual_net_eur_ha.idxmax()]
        summary.append(dict(scenario=name, operation=mode,
                            capture_price_ct=conv.capture_price_ct.median(),
                            best_conv_net=best_inc.annual_net_eur_ha, best_conv_rows=int(best_inc.n_rows),
                            max_LER_conv_net=conv.loc[conv.LER.idxmax()].annual_net_eur_ha,
                            conv_positive=int((conv.annual_net_eur_ha > 0).sum()), n_conv=len(conv),
                            best_semi_net=g[g.pv_type == "semi_transparent"].annual_net_eur_ha.max(),
                            pareto_jaccard=len(cur & base_set)/len(cur | base_set)))
    pd.concat(parts).to_csv(folder/"analysis"/"price_scenarios.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(folder/"analysis"/"price_scenarios_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recheck_combined_192skies")
