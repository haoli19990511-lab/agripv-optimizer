"""Crop comparison on the re-checked Pareto designs (Section 4.x): same geometries, cached physics, other crops.

    python analysis/crop_comparison.py
Reads the re-check folders of winter wheat (no suffix) and the crop runs (suffix _<crop>_<variety>) for the three
mounting types, conventional modules only. Writes results/analysis_crops/: crop_summary.csv, crop_designs.csv,
fig_crop_comparison.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS, mark_pareto

BASE = {"Fixed tilt": "recheck_combined_192skies", "Single-axis": "single_axis_recheck_combined_192skies",
        "Vertical": "vertical_recheck_combined_192skies"}
CROPS = {"Winter wheat": "", "Spring barley": "_barley_Spring_barley_301", "Grain maize": "_maize_Grain_maize_201",
         "Potato": "_potato_Potato_701", "Sugar beet": "_sugarbeet_Sugarbeet_601", "Soybean": "_soybean_Soybean_901"}
COL = {"Winter wheat": "#d4a017", "Spring barley": "#9467bd", "Grain maize": "#2ca02c", "Potato": "#8c564b",
       "Sugar beet": "#d62728", "Soybean": "#1f77b4"}
OUT = RESULTS/"analysis_crops"
OUT.mkdir(exist_ok=True)

rows, designs = [], []
for crop, suffix in CROPS.items():
    for mounting, base in BASE.items():
        f = RESULTS/(base+suffix)/"evaluated_candidates.csv"
        if not f.exists():
            print("missing", f.parent.name)
            continue
        t = pd.read_csv(f)
        t = t[t.pv_type == "conventional"].copy()
        t["feasible"] = t.feasible.astype(bool)
        t["crop"], t["mounting"] = crop, mounting
        feas = mark_pareto(t[t.feasible].copy()) if t.feasible.any() else t.iloc[:0]
        designs.append(t)
        best = feas.loc[feas.annual_net_eur_ha.idxmax()] if len(feas) else None
        rows.append(dict(crop=crop, mounting=mounting, n=len(t), n_feasible=int(t.feasible.sum()),
                         immature=int((~t.crop_all_points_mature.astype(bool)).sum()),
                         below_66=int((t.LER_crop < .66).sum()),
                         open_yield_t_ha=t.natural_yield_t_ha.iloc[0],
                         LER_crop=[t.LER_crop.min(), t.LER_crop.max()], LER=[t.LER.min(), t.LER.max()],
                         crop_revenue_share=float((t.crop_revenue_eur/(t.crop_revenue_eur+t.electrical_revenue_eur)).median()),
                         best_net=None if best is None else best.annual_net_eur_ha,
                         best_rows=None if best is None else int(best.n_rows),
                         best_LER=None if best is None else best.LER, n_pareto=int(feas.pareto.sum()) if len(feas) else 0))
s = pd.DataFrame(rows)
s.to_csv(OUT/"crop_summary.csv", index=False)
d = pd.concat(designs)
d.to_csv(OUT/"crop_designs.csv", index=False)

# Relative yield of the same designs, crop versus wheat
w = d[d.crop == "Winter wheat"].set_index(["mounting", "design_id"]).LER_crop
rel = {}
for crop in CROPS:
    if crop == "Winter wheat":
        continue
    c = d[d.crop == crop].set_index(["mounting", "design_id"]).LER_crop
    j = pd.concat([w, c], axis=1, keys=["wheat", "crop"]).dropna()
    rel[crop] = dict(mean_diff=float((j.crop-j.wheat).mean()), min_diff=float((j.crop-j.wheat).min()),
                     max_diff=float((j.crop-j.wheat).max()), r=float(j.corr().iloc[0, 1]))
(OUT/"crop_vs_wheat.json").write_text(json.dumps(rel, indent=1))
pd.set_option("display.width", 250, "display.max_columns", 30)
print(s.round(3).to_string(index=False))
print(json.dumps(rel, indent=1))

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.3), sharey=True)
for ax, mounting in zip(axes, BASE):
    for crop in CROPS:
        g = d[(d.crop == crop) & (d.mounting == mounting)]
        if g.empty:
            continue
        ax.scatter(g.LER_pv, g.LER_crop, s=16, color=COL[crop], alpha=.85, lw=0, label=crop)
    ax.axhline(.66, color="#52514e", lw=.8, ls="--")
    ax.set_title(mounting, fontsize=9)
    ax.set_xlabel("LER$_{PV}$ (–)")
    ax.grid(color="#e6e5e0", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
axes[0].set_ylabel("LER$_{crop}$ (–)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=6, markerscale=1.5)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(OUT/"fig_crop_comparison.png", dpi=250)
print("Saved:", OUT)
