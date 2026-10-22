"""Thesis tables and quoted numbers from the final re-checks (corrected model, 192 skies for all mountings).

    python analysis/final_tables.py
Writes results/final_tables/ (file names = table numbers of the thesis): table04_fixed.csv, table05_by_rows.csv,
table06_vertical.csv, table08_monthly.csv, table09_sensitivity.csv, table10_price_years.csv,
table15_single_axis.csv (annex) and numbers.json (numbers quoted in the text)
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS, crf, load_run, mark_pareto, recompute_economics

FIXED, SINGLE, VERTICAL = "recheck_combined_192skies", "single_axis_recheck_combined_192skies", "vertical_recheck_combined_192skies"
OUT = RESULTS/"final_tables"
OUT.mkdir(exist_ok=True)
N = {}


def fmt(x, d=0):
    return f"{x:,.{d}f}".replace("-", "−")


def load(folder):
    t, meta, _ = load_run(folder)
    t = t[t.feasible.astype(bool)].copy()
    t["net_no_bonus"] = recompute_economics(t, dict(meta["config"], agripv_bonus_eur_kwh=0.)).annual_net_eur_ha
    t["kwh_kwp"] = t.electrical_kwh/t.installed_kwp
    t["capex_kwp"] = t.capex_total_eur/t.installed_kwp
    return t, meta["config"]


fixed, cfg_f = load(FIXED)
single, cfg_s = load(SINGLE)
vert, cfg_v = load(VERTICAL)
parts = []
for kind, g in fixed.groupby("pv_type"):
    parts.append(mark_pareto(g.copy()))
fixed = pd.concat(parts)
conv = fixed[fixed.pv_type == "conventional"]
semi = fixed[fixed.pv_type == "semi_transparent"]
single = mark_pareto(single[single.pv_type == "conventional"].copy())
vert = mark_pareto(vert[vert.pv_type == "conventional"].copy())

# Table 4: conventional fixed-tilt Pareto designs
p = conv[conv.pareto].sort_values("LER")
t4 = pd.DataFrame({"Rows": p.n_rows, "Tilt (°)": p.tilt.map(lambda v: f"{v:.1f}"), "Pitch (m)": p.pitch.map(lambda v: f"{v:.2f}"),
                   "Azim. (°)": p.azimuth.round().astype(int), "Clear. (m)": p.clearance_height.map(lambda v: f"{v:.2f}"),
                   "LERcrop": p.LER_crop.map(lambda v: f"{v:.3f}"), "LERPV": p.LER_pv.map(lambda v: f"{v:.3f}"),
                   "LER": p.LER.map(lambda v: f"{v:.3f}"), "Net, bonus": p.annual_net_eur_ha.map(fmt),
                   "Net, no bonus": p.net_no_bonus.map(fmt)})
t4.to_csv(OUT/"table04_fixed.csv", index=False)

# Table 5: best net income per number of rows, fixed (F) and single-axis (S)
rows = []
for n in sorted(set(conv.n_rows) | set(single.n_rows)):
    for tag, g in (("F", conv), ("S", single)):
        g = g[g.n_rows == n]
        if g.empty:
            continue
        b = g.loc[g.annual_net_eur_ha.idxmax()]
        rows.append({"Rows": f"{n} {tag}", "Pitch (m)": f"{b.pitch:.2f}", "LERcrop": f"{b.LER_crop:.3f}",
                     "LERPV": f"{b.LER_pv:.3f}", "LER": f"{b.LER:.3f}", "kWh/kWp": fmt(b.kwh_kwp),
                     "Price (ct/kWh)": f"{b.capture_price_eur_kwh*100:.2f}", "CAPEX (EUR/kWp)": fmt(b.capex_kwp),
                     "Net income": fmt(b.annual_net_eur_ha), "_kwh": b.kwh_kwp, "_price": b.capture_price_eur_kwh*100,
                     "_capex": b.capex_kwp, "_net": b.annual_net_eur_ha, "_n": n, "_tag": tag})
t5 = pd.DataFrame(rows)
t5[[c for c in t5.columns if not c.startswith("_")]].to_csv(OUT/"table05_by_rows.csv", index=False)
pairs = t5.pivot(index="_n", columns="_tag", values=["_kwh", "_price", "_capex", "_net"]).dropna()
N["tracker_kwh_gain_pct"] = [float(((pairs["_kwh"]["S"]/pairs["_kwh"]["F"])-1).min()*100), float(((pairs["_kwh"]["S"]/pairs["_kwh"]["F"])-1).max()*100)]
N["tracker_price"] = [float(pairs["_price"]["S"].min()), float(pairs["_price"]["S"].max())]
N["fixed_price"] = [float(pairs["_price"]["F"].min()), float(pairs["_price"]["F"].max())]
rev = (pairs["_kwh"]["S"]*pairs["_price"]["S"])/(pairs["_kwh"]["F"]*pairs["_price"]["F"])-1
N["tracker_rev_gain_pct"] = [float(rev.min()*100), float(rev.max()*100)]
cap = pairs["_capex"]["S"]/pairs["_capex"]["F"]-1
N["tracker_capex_pct"] = [float(cap.min()*100), float(cap.max()*100)]
N["tracker_gap_by_rows"] = {int(k): float(v) for k, v in (pairs["_net"]["F"]-pairs["_net"]["S"]).items()}

# Table 6: vertical Pareto designs with structure cost variants and break-even
p = vert[vert.pareto].sort_values("LER").copy()
for s in (100, 150):
    p[f"net{s}"] = recompute_economics(p, dict(cfg_v, vertical_structure_eur_kwp=float(s))).annual_net_eur_ha
slope = (p.net100-p.annual_net_eur_ha)/50.
p["break_even"] = 50.-p.annual_net_eur_ha/slope
t6 = pd.DataFrame({"Rows": p.n_rows, "Pitch (m)": p.pitch.map(lambda v: f"{v:.2f}"), "Azim. (°)": p.azimuth.round().astype(int),
                   "Clear. (m)": p.clearance_height.map(lambda v: f"{v:.2f}"), "LERcrop": p.LER_crop.map(lambda v: f"{v:.3f}"),
                   "LERPV": p.LER_pv.map(lambda v: f"{v:.3f}"), "LER": p.LER.map(lambda v: f"{v:.3f}"),
                   "Net, 50": p.annual_net_eur_ha.map(fmt), "Net, 100": p.net100.map(fmt), "Net, 150": p.net150.map(fmt),
                   "Break-even (EUR/kWp)": p.break_even.round().astype(int)})
t6.to_csv(OUT/"table06_vertical.csv", index=False)
best_v = p.loc[p.annual_net_eur_ha.idxmax()]
N["vertical"] = dict(n=len(vert), n_pareto=int(vert.pareto.sum()), LER=[p.LER.min(), p.LER.max()],
                     positive=int((p.annual_net_eur_ha > 0).sum()), best=best_v.annual_net_eur_ha, best_LER=best_v.LER,
                     best_break_even=best_v.break_even, best_net100=best_v.net100, best_net150=best_v.net150,
                     three_row_break_even=p[p.n_rows == 3].break_even.round().tolist(), azimuth=[p.azimuth.min(), p.azimuth.max()],
                     clearance=[p.clearance_height.min(), p.clearance_height.max()], kwh_kwp=[vert.kwh_kwp.min(), vert.kwh_kwp.max()],
                     price=[p.capture_price_eur_kwh.min()*100, p.capture_price_eur_kwh.max()*100],
                     pitches={int(n): [g.pitch.min(), g.pitch.max()] for n, g in p.groupby("n_rows")},
                     LER_by_rows={int(n): [g.LER.min(), g.LER.max()] for n, g in p.groupby("n_rows")},
                     cropped_share=[float(1-0.5*n/24) for n in (p.n_rows.max(), p.n_rows.min())])

# Table 12: single-axis Pareto designs
p = single[single.pareto].sort_values("LER")
t12 = pd.DataFrame({"Rows": p.n_rows, "Pitch (m)": p.pitch.map(lambda v: f"{v:.2f}"), "Axis az. (°)": p.axis_azimuth.round().astype(int),
                    "Hub (m)": p.hub_height.map(lambda v: f"{v:.2f}"), "Limit (°)": p.limit_angle.map(lambda v: f"{v:.1f}"),
                    "Lowest edge (m)": p.min_clearance_m.map(lambda v: f"{v:.2f}"), "LERcrop": p.LER_crop.map(lambda v: f"{v:.3f}"),
                    "LERPV": p.LER_pv.map(lambda v: f"{v:.3f}"), "LER": p.LER.map(lambda v: f"{v:.3f}"),
                    "Net, bonus": p.annual_net_eur_ha.map(fmt), "Net, no bonus": p.net_no_bonus.map(fmt)})
t12.to_csv(OUT/"table15_single_axis.csv", index=False)
N["single"] = dict(n=len(single), n_pareto=int(single.pareto.sum()), LER=[p.LER.min(), p.LER.max()],
                   best=single.annual_net_eur_ha.max(), best_no_bonus=single.net_no_bonus.max(),
                   hub=[p.hub_height.min(), p.hub_height.max()], edge=[p.min_clearance_m.min(), p.min_clearance_m.max()],
                   limit=sorted(p.limit_angle.round(1).tolist()), axis=[p.axis_azimuth.min(), p.axis_azimuth.max()],
                   pitches={int(n): [g.pitch.min(), g.pitch.max()] for n, g in p.groupby("n_rows")},
                   max_LER=single.LER.max())

# Fixed-tilt numbers
pc, ps = conv[conv.pareto], semi[semi.pareto]
best = conv.loc[conv.annual_net_eur_ha.idxmax()]
top = conv.loc[conv.LER.idxmax()]
five = conv[(conv.n_rows == 5)]
cheap5 = five.loc[five.annual_net_eur_ha.idxmax()]
N["fixed"] = dict(n_pareto_conv=len(pc), conv_LER=[pc.LER.min(), pc.LER.max()], best=best.annual_net_eur_ha,
                  best_pitch=best.pitch, best_LER=best.LER, best_rows=int(best.n_rows), top_LER=top.LER,
                  top_net=top.annual_net_eur_ha, cheap5_LER=cheap5.LER, cheap5_net=cheap5.annual_net_eur_ha,
                  rows_on_front=sorted(pc.n_rows.unique().tolist()),
                  pitches={int(n): [g.pitch.min(), g.pitch.max()] for n, g in pc.groupby("n_rows")},
                  LERpv_by_rows={int(n): [g.LER_pv.min(), g.LER_pv.max()] for n, g in pc.groupby("n_rows")},
                  LERcrop_by_rows={int(n): [g.LER_crop.min(), g.LER_crop.max()] for n, g in pc.groupby("n_rows")},
                  clear_by_rows5=[pc[pc.n_rows == 5].clearance_height.min(), pc[pc.n_rows == 5].clearance_height.max()],
                  tilt=[pc.tilt.min(), pc.tilt.max()], azimuth=[pc.azimuth.min(), pc.azimuth.max()],
                  clearance_min=pc.clearance_height.min(), kwh_kwp_by_rows={int(n): [g.kwh_kwp.min(), g.kwh_kwp.max()] for n, g in pc.groupby("n_rows")},
                  bonus=[(conv.annual_net_eur_ha-conv.net_no_bonus).min(), (conv.annual_net_eur_ha-conv.net_no_bonus).max()],
                  positive_bonus=int((conv.annual_net_eur_ha > 0).sum()), positive_rows=sorted(conv[conv.annual_net_eur_ha > 0].n_rows.unique().tolist()),
                  best_no_bonus=conv.net_no_bonus.max(), n_pareto_semi=len(ps), semi_LER=[ps.LER.min(), ps.LER.max()],
                  semi_net=[ps.annual_net_eur_ha.min(), ps.annual_net_eur_ha.max()],
                  semi_kwh_kwp=[semi.kwh_kwp.min(), semi.kwh_kwp.max()], conv_kwh_kwp=[conv.kwh_kwp.min(), conv.kwh_kwp.max()],
                  semi_capex=[semi.capex_kwp.min(), semi.capex_kwp.max()], conv_capex=[conv.capex_kwp.min(), conv.capex_kwp.max()],
                  LERcrop_min=conv.LER_crop.min())
j = conv.set_index("design_id").join(semi.set_index("design_id"), rsuffix="_s", how="inner")
N["fixed"]["semi_minus_conv"] = dict(LERcrop=[(j.LER_crop_s-j.LER_crop).min(), (j.LER_crop_s-j.LER_crop).max()],
                                     LERpv=[(j.LER_pv_s-j.LER_pv).min(), (j.LER_pv_s-j.LER_pv).max()],
                                     kwh_kwp=[(j.kwh_kwp_s/j.kwh_kwp-1).min()*100, (j.kwh_kwp_s/j.kwh_kwp-1).max()*100],
                                     kwp=[(j.installed_kwp_s/j.installed_kwp-1).mean()*100],
                                     field_area=[(j.field_area_ha_s/j.field_area_ha-1).mean()*100])
both = pd.concat([conv, semi])
both = mark_pareto(both.copy())
N["fixed"]["joint_semi_on_front"] = int(both[both.pareto & (both.pv_type == "semi_transparent")].shape[0])
N["fixed"]["joint_semi_LER"] = [both[both.pareto & (both.pv_type == "semi_transparent")].LER.min(),
                                both[both.pareto & (both.pv_type == "semi_transparent")].LER.max()]
N["fixed"]["joint_semi_net"] = [both[both.pareto & (both.pv_type == "semi_transparent")].annual_net_eur_ha.min(),
                                both[both.pareto & (both.pv_type == "semi_transparent")].annual_net_eur_ha.max()]

# Re-check versus search
for name, folder in (("fixed", FIXED), ("single", SINGLE), ("vertical", VERTICAL)):
    c = pd.read_csv(RESULTS/folder/"comparison_with_search.csv")
    c = c[c.pv_type == "conventional"] if name != "fixed" else c
    N[f"recheck_{name}"] = {k: [c[f"{k}_change_%"].min(), c[f"{k}_change_%"].max()]
                            for k in ("electrical_kwh", "crop_yield_t_ha", "LER", "LER_pv")}

# Joint front over mountings
allc = pd.concat([conv.assign(m="fixed"), single.assign(m="single"), vert.assign(m="vertical")], ignore_index=True)
allc = mark_pareto(allc.drop(columns="pareto"))
jf = allc[allc.pareto].sort_values("LER")
N["joint_front"] = jf[["m", "n_rows", "LER", "annual_net_eur_ha"]].round(3).values.tolist()

# Tables 7-9 from the existing analysis outputs
A = RESULTS/FIXED/"analysis"
m = pd.read_csv(A/"monthly_capture_price.csv")
m = m[m.design == "highest income"]
t7 = pd.DataFrame({"Month": m.month, "Share of annual yield (%)": (m.energy_share*100).map(lambda v: f"{v:.1f}"),
                   "PV-weighted price (ct/kWh)": m.capture_price_ct_kwh.map(lambda v: f"{v:.1f}"),
                   "Average price (ct/kWh)": m.average_price_ct_kwh.map(lambda v: f"{v:.1f}")})
t7.to_csv(OUT/"table08_monthly.csv", index=False)
mm = m[m.month != "year"].copy()
mm["month"] = mm.month.astype(int)
N["monthly"] = dict(summer_share=float(mm[mm.month.between(4, 9)].energy_share.sum()*100),
                    summer_price=[float(mm[mm.month.between(4, 9)].capture_price_ct_kwh.min()), float(mm[mm.month.between(4, 9)].capture_price_ct_kwh.max())],
                    year=m[m.month == "year"].iloc[0][["capture_price_ct_kwh", "average_price_ct_kwh", "negative_price_energy_share"]].astype(float).to_dict())

oat = pd.read_csv(A/"sensitivity_oat.csv")
oat = oat[oat.pv_type == "conventional"]
order = [("electricity_price_factor", "Electricity price", "× 0.7 / × 1.3"), ("discount_rate", "Discount rate", "3 % / 8 %"),
         ("other_capex_eur_kwp", "Other CAPEX", "500 / 800 EUR/kWp"), ("structure_base_eur_kwp", "Structure, base rate", "200 / 450 EUR/kWp"),
         ("lifetime_years", "Lifetime", "20 / 30 a"), ("opex_fraction", "O&M cost", "0.5 % / 2 % of CAPEX"),
         ("module_conv_eur_kwp", "Module price", "150 / 300 EUR/kWp"), ("crop_price_eur_t", "Wheat price", "180 / 280 EUR/t"),
         ("crop_cost_eur_planted_ha", "Crop production cost", "800 / 1,300 EUR/ha"),
         ("structure_extra_eur_kwp_m", "Structure, per m above 3 m", "75 / 300 EUR/kWp")]
rows = []
for key, label, rng in order:
    g = oat[oat.parameter == key].set_index(["design", "side"]).annual_net_eur_ha
    rows.append({"Parameter": label, "Low / high": rng, "Highest income, low": fmt(g["max net income", "low"]),
                 "Highest income, high": fmt(g["max net income", "high"]), "Highest LER, low": fmt(g["max LER", "low"]),
                 "Highest LER, high": fmt(g["max LER", "high"])})
pd.DataFrame(rows).to_csv(OUT/"table09_sensitivity.csv", index=False)
N["sensitivity_jaccard_min"] = float(pd.read_csv(A/"sensitivity_oat.csv").pareto_jaccard.min())
e = oat[oat.parameter == "electricity_price_factor"].set_index(["design", "side"]).annual_net_eur_ha
slope = (e["max net income", "high"]-e["max net income", "low"])/0.6
N["break_even_price_factor_best"] = float(1-best.annual_net_eur_ha/slope)
N["break_even_price_ct_best"] = N["break_even_price_factor_best"]*best.capture_price_eur_kwh*100
N["best_capture_ct"] = best.capture_price_eur_kwh*100
be = pd.read_csv(A/"break_even_module_price.csv").set_index("design_id")
N["break_even_module_best"] = float(be.loc[best.design_id, "break_even_module_eur_kwp"].iloc[0]
                                    if isinstance(be.loc[best.design_id, "break_even_module_eur_kwp"], pd.Series)
                                    else be.loc[best.design_id, "break_even_module_eur_kwp"])
bs = semi.loc[semi.annual_net_eur_ha.idxmax()]
semi_slope = bs.electrical_revenue_eur/bs.footprint_ha  # net income change per unit price factor
N["semi_break_even_price_ct"] = float((1-bs.annual_net_eur_ha/semi_slope)*bs.capture_price_eur_kwh*100)
bes = be[be.pv_type == "semi_transparent"]
N["semi_break_even_module_best"] = float(bes.loc[bs.design_id, "break_even_module_eur_kwp"])
N["semi_break_even_module_pareto"] = bes[bes.pareto].break_even_module_eur_kwp.describe().round(1).to_dict()

ps_ = pd.read_csv(A/"price_scenarios_summary.csv")
names = {"2023-2025 (base)": "2023–2025 (mean, base)", "2020-2022": "2020–2022", "2023": "2023", "2024": "2024", "2025": "2025"}
t9 = pd.DataFrame({"Prices": ps_.scenario.map(names), "Feed-in": ps_.operation.map({"feed-in always": "always", "curtailed at p<0": "curtailed at p < 0"}),
                   "PV-weighted price (ct/kWh)": ps_.capture_price_ct.map(lambda v: f"{v:.1f}"),
                   "Best conventional (rows)": [f"{fmt(a)} ({int(b)})" for a, b in zip(ps_.best_conv_net, ps_.best_conv_rows)],
                   "Highest-LER conventional": ps_.max_LER_conv_net.map(fmt),
                   "Profitable conventional": [f"{int(a)} / {int(b)}" for a, b in zip(ps_.conv_positive, ps_.n_conv)],
                   "Best semi-transparent": ps_.best_semi_net.map(fmt)})
t9.to_csv(OUT/"table10_price_years.csv", index=False)
N["price_years"] = ps_.round(3).to_dict(orient="records")

(OUT/"numbers.json").write_text(json.dumps(N, indent=1, default=float))
print(json.dumps(N, indent=1, default=lambda v: round(float(v), 4)))
for f in sorted(OUT.glob("table*.csv")):
    print("\n==", f.name)
    print(pd.read_csv(f, dtype=str).to_string(index=False))
