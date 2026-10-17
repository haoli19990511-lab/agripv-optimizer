"""Summarise the final re-checks (corrected model) next to the draft results, per mounting and construction.

Usage (agripv env, optimizer folder): python analysis/final_summary.py
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]/"results"
RUNS = {  # mounting: (final run, draft run)
    "fixed": ("recheck_combined_192skies", "recheck_combined_144skies"),
    "single_axis": ("single_axis_recheck_combined_192skies", "single_axis_recheck_combined_192skies_draft"),
    "vertical": ("vertical_recheck_combined_192skies", "vertical_recheck_combined_192skies_draft"),
}
pd.set_option("display.width", 250, "display.max_columns", 30)


def no_bonus(df):
    ha = df["field_area_ha"]
    return df["annual_net_eur_ha"] - df["agripv_bonus_eur"]/ha


def pareto(df, cols=("LER", "annual_net_eur_ha")):
    a, b = cols
    keep = []
    for i, r in df.iterrows():
        dom = ((df[a] >= r[a]) & (df[b] >= r[b]) & ((df[a] > r[a]) | (df[b] > r[b]))).any()
        keep.append(not dom)
    return df[keep]


rows = []
for mounting, (final, draft) in RUNS.items():
    for label, run in (("final", final), ("draft", draft)):
        f = R/run/"evaluated_candidates.csv"
        if not f.exists():
            continue
        ev = pd.read_csv(f)
        ev = ev[ev["feasible"]]
        for kind, g in ev.groupby("pv_type"):
            front = pareto(g)
            best = g.loc[g["annual_net_eur_ha"].idxmax()]
            top = g.loc[g["LER"].idxmax()]
            rows.append(dict(mounting=mounting, run=label, pv_type=kind, n=len(g), n_front=len(front),
                LER_min=front["LER"].min(), LER_max=front["LER"].max(),
                best_net=best["annual_net_eur_ha"], best_net_nobonus=no_bonus(g).max(), best_rows=best["n_rows"],
                best_LER=best["LER"], best_LERcrop=best["LER_crop"], best_LERpv=best["LER_pv"],
                kwh_kwp_min=(g["electrical_kwh"]/g["installed_kwp"]).min(),
                kwh_kwp_max=(g["electrical_kwh"]/g["installed_kwp"]).max(),
                price=best["capture_price_eur_kwh"]*100, n_profitable=int((g["annual_net_eur_ha"] > 0).sum()),
                maxLER_net=top["annual_net_eur_ha"], LERcrop_min=g["LER_crop"].min()))
out = pd.DataFrame(rows)
print(out.round(3).to_string())
out.to_csv(R/"final_summary.csv", index=False)

# Semi-transparent vs conventional for the same geometry (fixed tilt, final)
ev = pd.read_csv(R/RUNS["fixed"][0]/"evaluated_candidates.csv")
c = ev[ev.pv_type == "conventional"].set_index("design_id")
s = ev[ev.pv_type == "semi_transparent"].set_index("design_id")
j = c.join(s, lsuffix="_c", rsuffix="_s", how="inner")
print("\nsemi - conv, same geometry (fixed, final):")
print(pd.DataFrame(dict(dLERcrop=j.LER_crop_s - j.LER_crop_c, dLERpv=j.LER_pv_s - j.LER_pv_c,
    kwh_kwp_c=j.electrical_kwh_c/j.installed_kwp_c, kwh_kwp_s=j.electrical_kwh_s/j.installed_kwp_s,
    capex_kwp_c=j.capex_total_eur_c/j.installed_kwp_c, capex_kwp_s=j.capex_total_eur_s/j.installed_kwp_s)).describe().round(3))
# Joint front over mountings (conventional, final)
allc = []
for mounting, (final, _) in RUNS.items():
    ev = pd.read_csv(R/final/"evaluated_candidates.csv")
    ev = ev[(ev.pv_type == "conventional") & ev.feasible]
    ev["mounting"] = mounting
    allc.append(ev)
allc = pd.concat(allc)
jf = pareto(allc).sort_values("LER")
print("\njoint front (conventional, final):")
print(jf[["mounting", "n_rows", "pitch", "LER_crop", "LER_pv", "LER", "annual_net_eur_ha"]].round(3).to_string())
