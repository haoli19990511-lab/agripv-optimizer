"""Literature-based cost scenarios (no physics re-run): Pareto fronts, net income and the
electricity price each design needs to break even.
    python analysis/cost_scenarios.py nsga2_combined
Outputs in results/<run>/analysis/: cost_scenarios.csv, cost_scenarios_summary.csv,
fig_cost_scenarios.png

Sources (cite in the thesis):
- SYMBIOSYST D7.3 (2026), p.13: structures ~300 EUR/kWp for horticulture (< 5 m) and ~600 EUR/kWp for
  arable crops (> 5 m); site preparation 150-300 EUR/kWp; modules 150-750 EUR/kWp; arable agri-PV
  total CAPEX ~1500 +/- 425 EUR/kWp (ground PV ~650).
- Bellone et al. (2024) Applied Energy 369:123490, Table 10: inverters 58-77, electrical BOS 100-282,
  installation and work 227-373, professionals and fees ~61-77 EUR/kWp (sum ~500-600).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, LABELS, crf, load_run, mark_pareto, pareto_mask, recompute_economics

# Structure: 300 EUR/kWp at 3 m rising 150 EUR/kWp per m -> 600 EUR/kWp at 5 m (D7.3 range).
LIT = dict(structure_base_eur_kwp=300., structure_reference_height_m=3., structure_extra_eur_kwp_m=150.)
SCENARIOS = {
    "S0 as run (cost config of this run)": {},
    "S1 literature, semi 650": dict(LIT, other_capex_eur_kwp=650., module_semi_eur_kwp=650.),
    "S5 v6 costs (module + structure only)": dict(structure_base_eur_kwp=600., other_capex_eur_kwp=0.),
    "S2 literature, semi 450": dict(LIT, other_capex_eur_kwp=650., module_semi_eur_kwp=450.),
    "S3 literature, semi 300": dict(LIT, other_capex_eur_kwp=650., module_semi_eur_kwp=300.),
    "S4 literature high (site prep 300)": dict(LIT, other_capex_eur_kwp=800., module_semi_eur_kwp=650.),
}


def break_even_price_factor(t, cfg):
    """Factor on the hourly price profile at which annual net income is zero."""
    s = recompute_economics(t, cfg, 1.0)
    non_elec = s.annual_net_eur - s.electrical_revenue_eur
    return -non_elec/s.electrical_revenue_eur


def main(run):
    table, meta, folder = load_run(run)
    if "config" not in meta:
        meta["config"] = load_run(f"nsga2_{meta.get('search_sampling', {}).get('mode', 'combined')}")[1]["config"]
    out = folder/"analysis"
    out.mkdir(exist_ok=True)
    base = dict(meta["config"])
    # Average realised price of the base profile, EUR/kWh, to express the break-even as a price.
    avg_price = (table.electrical_revenue_eur/table.electrical_kwh).median()

    rows, summary = [], []
    for name, change in SCENARIOS.items():
        cfg = dict(base, **change)
        t = mark_pareto(recompute_economics(table, cfg))
        t["scenario"] = name
        t["capex_eur_kwp"] = t.capex_total_eur/t.installed_kwp
        t["break_even_price_eur_kwh"] = break_even_price_factor(table, cfg)*avg_price
        f = t[t.feasible.astype(bool)]
        joint = f[pareto_mask(f[["LER", "annual_net_eur_ha"]])]
        t["joint_pareto"] = t.index.isin(joint.index)
        rows.append(t)
        for kind, g in f.groupby("pv_type"):
            p = g[g.pareto]
            summary.append(dict(scenario=name, pv_type=kind,
                                capex_eur_kwp_range=f"{g.capex_eur_kwp.min():.0f}-{g.capex_eur_kwp.max():.0f}",
                                net_min=g.annual_net_eur_ha.min(), net_max=g.annual_net_eur_ha.max(),
                                share_positive=(g.annual_net_eur_ha > 0).mean(),
                                pareto_n=len(p), pareto_LER_max=p.LER.max(),
                                pareto_net_max=p.annual_net_eur_ha.max(),
                                min_break_even_price=g.break_even_price_eur_kwh.min(),
                                on_joint_front=int(t.joint_pareto[t.pv_type == kind].sum())))
    allrows = pd.concat(rows, ignore_index=True)
    allrows.to_csv(out/"cost_scenarios.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(out/"cost_scenarios_summary.csv", index=False)
    print(f"Median realised electricity price of the base profile: {avg_price:.4f} EUR/kWh")
    print(s.round(3).to_string(index=False))
    plot(allrows, out)
    print("Saved:", out)


def plot(t, out):
    names = list(SCENARIOS)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3*len(names), 3.4), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        s = t[t.scenario == name]
        for kind in ("conventional", "semi_transparent"):
            g = s[s.pv_type == kind]
            p = g[g.pareto].sort_values("LER")
            ax.scatter(g.LER, g.annual_net_eur_ha/1000, s=8, color=COLORS[kind], alpha=.2, lw=0)
            ax.plot(p.LER, p.annual_net_eur_ha/1000, color=COLORS[kind], lw=1.8, marker="o", ms=4,
                    label=LABELS[kind])
        ax.axhline(0, color="#52514e", lw=.8)
        ax.set_title(name, fontsize=8)
        ax.grid(color="#e6e5e0", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("LER", fontsize=8)
    axes[0].set_ylabel("Annual net income (k EUR/ha/yr)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"fig_cost_scenarios.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nsga2_combined")
