"""Compare the re-checked Pareto fronts of the mounting types (conventional modules, agri-PV bonus).
    python analysis/compare_mountings.py
Uses whichever re-check folders exist. Writes to results/analysis_mountings/:
  fig_mountings_pareto.png   Pareto fronts per mounting type
  mountings_pareto.csv       Pareto designs per mounting type (with net income without bonus)
  mountings_by_rows.csv      best-income design per mounting type and row count
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS, check_reproduction, load_run, mark_pareto, recompute_economics

RUNS = {"Fixed tilt": "recheck_combined_144skies", "Single-axis": "single_axis_recheck_combined_192skies",
        "Vertical": "vertical_recheck_combined_192skies"}
COL = {"Fixed tilt": "#2a78d6", "Single-axis": "#6b5bb5", "Vertical": "#2f8a4f"}
MARK = {"Fixed tilt": "o", "Single-axis": "D", "Vertical": "^"}


def load():
    parts = []
    for name, folder in RUNS.items():
        if not (RESULTS/folder/"evaluated_candidates.csv").exists():
            continue
        t, meta, _ = load_run(folder)
        check_reproduction(t, meta["config"])
        t = t[t.pv_type == "conventional"].copy()
        t["net_no_bonus_eur_ha"] = recompute_economics(t, dict(meta["config"], agripv_bonus_eur_kwh=0.)).annual_net_eur_ha
        t["kwh_per_kwp"] = t.electrical_kwh/t.installed_kwp
        t["capex_eur_kwp"] = t.capex_total_eur/t.installed_kwp
        t["mounting_type"] = name
        parts.append(mark_pareto(t))
    return pd.concat(parts, ignore_index=True)


def main():
    out = RESULTS/"analysis_mountings"
    out.mkdir(exist_ok=True)
    t = load()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, g in t.groupby("mounting_type", sort=False):
        ax.scatter(g.LER, g.annual_net_eur_ha/1000, s=10, color=COL[name], alpha=.25, lw=0)
        p = g[g.pareto].sort_values("LER")
        ax.plot(p.LER, p.annual_net_eur_ha/1000, color=COL[name], lw=2, marker=MARK[name], ms=5.5, mec="white",
                mew=1.1, label=f"{name} (n={len(p)})")
    ax.axhline(0, color="#52514e", lw=.8)
    ax.set_xlabel("Land equivalent ratio LER (–)")
    ax.set_ylabel("Annual net income (k EUR/(ha·a))")
    ax.grid(color="#e6e5e0", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out/"fig_mountings_pareto.png", dpi=250)
    plt.close(fig)

    keep = ["mounting_type", "design_id", "n_rows", "pitch", "tilt", "azimuth", "clearance_height", "axis_azimuth",
            "hub_height", "limit_angle", "min_clearance_m", "clear_corridor_m", "LER_crop", "LER_pv", "LER",
            "kwh_per_kwp", "capex_eur_kwp", "capture_price_eur_kwh", "annual_net_eur_ha", "net_no_bonus_eur_ha"]
    p = t[t.pareto][[c for c in keep if c in t.columns]].sort_values(["mounting_type", "LER"])
    p.round(4).to_csv(out/"mountings_pareto.csv", index=False)
    best = t[t.feasible.astype(bool)].sort_values("annual_net_eur_ha").groupby(["mounting_type", "n_rows"]).tail(1)
    best = best[[c for c in keep if c in t.columns]].sort_values(["n_rows", "mounting_type"])
    best.round(4).to_csv(out/"mountings_by_rows.csv", index=False)
    summary = {name: dict(n=int(len(g)), n_pareto=int(g.pareto.sum()), LER=[g[g.pareto].LER.min(), g[g.pareto].LER.max()],
                          net=[g[g.pareto].annual_net_eur_ha.min(), g[g.pareto].annual_net_eur_ha.max()],
                          best_no_bonus=g.net_no_bonus_eur_ha.max(), positive=int((g.annual_net_eur_ha > 0).sum()),
                          kwh_per_kwp=[g.kwh_per_kwp.min(), g.kwh_per_kwp.max()],
                          capex_eur_kwp=[g.capex_eur_kwp.min(), g.capex_eur_kwp.max()])
               for name, g in t.groupby("mounting_type", sort=False)}
    (out/"mountings_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=1, default=float))
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
