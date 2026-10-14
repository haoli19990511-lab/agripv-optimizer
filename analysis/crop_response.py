"""Crop response to shading (Section 4.4), from the stored WOFOST runs (no physics, no crop re-run).
    python analysis/crop_response.py
Each WOFOST run in runs/aachen_research stores its yield, transpiration and the daily irradiation it was driven
with. Runs are matched to designs by their yield, as in the results tables.
Writes results/analysis_crop_response/: crop_response_runs.csv, crop_response_summary.json,
fig_crop_response.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, GREY, LABELS, RESULTS, ROOT

DM = 0.86
TABLES = {"search (24 skies)": "nsga2_combined", "final re-check (144 skies)": "recheck_combined_144skies"}


def load_runs():
    rows = []
    for f in (ROOT/"runs"/"aachen_research").glob("*/Hao_crop_yield.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("model") != "Wofost72_PP" or d.get("yield_kg_DM_ha") is None:
            continue
        w = pd.read_csv(f.parent/"actual_model_weather.csv", usecols=["IRRAD"], encoding="utf-8-sig")
        rows.append(dict(scenario=d["scenario_id"], natural=d["scenario_id"] == "natural_light",
                         fresh=round(d["yield_kg_DM_ha"]/DM/1000., 9), transpiration_mm=d["model_transpiration_mm"],
                         irrad_MJ_m2=w.IRRAD.sum()/1e6, mature=d["all_points_mature"]))
    return pd.DataFrame(rows)


def main():
    out = RESULTS/"analysis_crop_response"
    out.mkdir(exist_ok=True)
    runs = load_runs()
    nat = runs[runs.natural]
    if nat.fresh.nunique() != 1:
        raise AssertionError("Open-field runs differ")
    nat = nat.iloc[0]
    apv = runs[~runs.natural].drop_duplicates("fresh")
    parts = []
    for label, folder in TABLES.items():
        t = pd.read_csv(RESULTS/folder/"evaluated_candidates.csv", float_precision="round_trip")
        t["fresh"] = t.crop_yield_t_ha.round(9)
        t = t.merge(apv[["fresh", "transpiration_mm", "irrad_MJ_m2"]], on="fresh", how="left")
        t["set"] = label
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["rel_yield"] = t.crop_yield_t_ha/t.natural_yield_t_ha
    t["rel_irradiation"] = t.irrad_MJ_m2/nat.irrad_MJ_m2
    t["rel_transpiration"] = t.transpiration_mm/nat.transpiration_mm
    m = t.dropna(subset=["irrad_MJ_m2"])
    cols = ["set", "pv_type", "design_id", "n_rows", "tilt", "pitch", "azimuth", "clearance_height", "rel_irradiation",
            "rel_yield", "rel_transpiration", "transpiration_mm", "crop_yield_t_ha", "LER_crop"]
    m[cols].to_csv(out/"crop_response_runs.csv", index=False)

    # The two sets are fitted separately: at equal seasonal irradiation the coarse search sampling gives lower
    # yields (Section 4.6), so a pooled fit would mix two relations.
    final = m[m.set.str.startswith("final")]
    search = m[m.set.str.startswith("search")]
    fit = np.polyfit(final.rel_irradiation, final.rel_yield, 1)
    fit_search = np.polyfit(search.rel_irradiation, search.rel_yield, 1)
    summary = dict(
        matched={k: int((m.set == k).sum()) for k in TABLES}, total={k: int((t.set == k).sum()) for k in TABLES},
        open_field_yield_t_ha=float(nat.fresh), open_field_irradiation_MJ_m2=float(nat.irrad_MJ_m2),
        open_field_transpiration_mm=float(nat.transpiration_mm),
        fit_slope=float(fit[0]), fit_intercept=float(fit[1]),
        fit_search_slope=float(fit_search[0]), fit_search_intercept=float(fit_search[1]),
        r_irr_yield_final=float(np.corrcoef(final.rel_irradiation, final.rel_yield)[0, 1]),
        r_irr_yield_search=float(np.corrcoef(search.rel_irradiation, search.rel_yield)[0, 1]),
        search_minus_final_at_0p85=float(np.polyval(fit_search, .85)-np.polyval(fit, .85)),
        r_transp_yield=float(np.corrcoef(m.rel_transpiration, m.rel_yield)[0, 1]),
        rel_irradiation_range=[float(m.rel_irradiation.min()), float(m.rel_irradiation.max())],
        rel_yield_range=[float(m.rel_yield.min()), float(m.rel_yield.max())],
        final_rel_irradiation_range=[float(final.rel_irradiation.min()), float(final.rel_irradiation.max())],
        final_rel_yield_range=[float(final.rel_yield.min()), float(final.rel_yield.max())],
        final_transpiration_mm=[float(final.transpiration_mm.min()), float(final.transpiration_mm.max())],
        final_rel_transpiration=[float(final.rel_transpiration.min()), float(final.rel_transpiration.max())],
        semi_minus_conv_rel_yield_same_design=float(
            final.pivot_table(index="design_id", columns="pv_type", values="rel_yield").diff(axis=1).iloc[:, -1].median()),
        elasticity_at_mean=float(fit[0]*final.rel_irradiation.mean()/final.rel_yield.mean()),
    )
    (out/"crop_response_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    for kind in ("conventional", "semi_transparent"):
        g = m[(m.pv_type == kind) & m.set.str.startswith("search")]
        ax.scatter(g.rel_irradiation, g.rel_yield, s=10, color=COLORS[kind], alpha=.3, lw=0,
                   label=f"{LABELS[kind]}, search (24 skies)")
        g = m[(m.pv_type == kind) & m.set.str.startswith("final")]
        ax.scatter(g.rel_irradiation, g.rel_yield, s=26, color=COLORS[kind], edgecolors="white", linewidths=.8,
                   label=f"{LABELS[kind]}, re-check (144 skies)")
    x = np.linspace(m.rel_irradiation.min(), 1, 20)
    ax.plot(x, np.polyval(fit, x), color="#52514e", lw=1, ls="--",
            label=f"Fit to re-check: y = {fit[0]:.2f}x + {fit[1]:.2f}")
    ax.plot([x[0], 1], [x[0], 1], color=GREY, lw=1, ls=":", label="1:1 line")
    ax.set_xlabel("Relative ground irradiation in the season (–)")
    ax.set_ylabel("Relative grain yield (–)")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    for kind in ("conventional", "semi_transparent"):
        g = m[m.pv_type == kind]
        ax.scatter(g.rel_transpiration, g.rel_yield, s=10, color=COLORS[kind], alpha=.5, lw=0, label=LABELS[kind])
    ax.set_xlabel("Relative potential transpiration (–)")
    ax.set_ylabel("Relative grain yield (–)")
    for a in axes:
        a.grid(color="#e6e5e0", lw=.6)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors="#52514e", labelsize=8)
    fig.tight_layout()
    fig.savefig(out/"fig_crop_response.png", dpi=250)
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    main()
