"""Sampling validation table with the NRW crop economics and the final 192-sky re-check added.

Reads results/sampling_validation/sampling_validation.csv (24/144/96 skies and the full year, written by
validate_sampling.py with the earlier crop assumptions), recomputes the crop economics with the NRW planning data
(no new physics), adds the same designs from the 192-sky re-check and writes errors against the full year.
    python analysis/validation_nrw.py
Output: results/sampling_validation/validation_nrw.csv and errors_vs_full_nrw.csv
"""
import json

import pandas as pd

from common import RESULTS, crf

PRICE, COST = 213.1, 1120.  # winter wheat, NRW planning data (see update_crop_economics.py)
out = RESULTS/"sampling_validation"
t = pd.read_csv(out/"sampling_validation.csv", float_precision="round_trip")
meta = json.loads((RESULTS/"recheck_combined_192skies"/"run_metadata.json").read_text(encoding="utf-8"))["config"]
annuity = crf(meta["discount_rate"], meta["lifetime_years"])
t["crop_revenue_eur"] = t.crop_yield_t_ha.fillna(0.)*t.planted_area_ha*PRICE
t["crop_cost_eur"] = t.planted_area_ha*COST
t["revenue_eur"] = t.electrical_revenue_eur + t.agripv_bonus_eur + t.crop_revenue_eur
t["annual_operating_cashflow_eur"] = t.revenue_eur - t.annual_opex_eur - t.crop_cost_eur
t["annual_net_eur"] = t.annual_operating_cashflow_eur - t.capex_total_eur*annuity
t["annual_net_eur_ha"] = t.annual_net_eur/t.footprint_ha

rc = pd.read_csv(RESULTS/"recheck_combined_192skies"/"evaluated_candidates.csv", float_precision="round_trip")
rc = rc[rc.pv_type == "conventional"].set_index("design_id")
names = t.drop_duplicates("design").set_index("design").design_id
add = rc.loc[names.values].reset_index()
add["design"] = names.index.values
add["sampling"], add["n_skies"] = "final re-check 2d x 8h", 192
add["kwh_per_kwp"] = add.electrical_kwh/add.installed_kwp
t = pd.concat([t, add[[c for c in t.columns if c in add.columns]]], ignore_index=True)
t.to_csv(out/"validation_nrw.csv", index=False)

metrics = ["kwh_per_kwp", "crop_yield_t_ha", "LER_pv", "LER_crop", "LER", "annual_net_eur_ha"]
full = t[t.sampling == "full year"].set_index("design")[metrics]
rows = []
for label, g in t.groupby("sampling", sort=False):
    g = g.set_index("design")[metrics]
    e = 100*(g/full - 1)
    e.columns = [f"{c}_error_%" for c in metrics]
    e["net_diff_eur_ha"] = g.annual_net_eur_ha - full.annual_net_eur_ha
    e["net_eur_ha"] = g.annual_net_eur_ha
    e["LER"] = g.LER
    e["sampling"], e["n_skies"] = label, int(t[t.sampling == label].n_skies.iloc[0])
    e["rank_LER"] = g.LER.rank(ascending=False).astype(int)
    e["rank_net"] = g.annual_net_eur_ha.rank(ascending=False).astype(int)
    rows.append(e.reset_index())
errors = pd.concat(rows, ignore_index=True).sort_values(["n_skies", "design"])
errors.to_csv(out/"errors_vs_full_nrw.csv", index=False)
pd.set_option("display.width", 250)
print(errors.round(2).to_string(index=False))
