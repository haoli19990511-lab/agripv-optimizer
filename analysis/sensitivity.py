"""One-at-a-time economic sensitivity, Pareto stability and break-even module price.

Runs in seconds on an existing results folder; no Radiance/WOFOST, no cache access.
    python analysis/sensitivity.py nsga2_combined
Outputs go to results/<run>/analysis/.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (COLORS, LABELS, check_reproduction, crf, load_run, mark_pareto,
                    recompute_economics, structure_rate, decisions)

# Low / high values per parameter. These are assumption ranges to justify in the
# thesis (cite sources); edit freely. "electricity_price_factor" scales all prices.
RANGES = {
    "electricity_price_factor": (0.7, 1.3),
    "module_conv_eur_kwp": (150., 300.),
    "module_semi_eur_kwp": (400., 900.),
    "structure_base_eur_kwp": (400., 900.),
    "structure_extra_eur_kwp_m": (50., 300.),
    "opex_fraction": (0.005, 0.02),
    "discount_rate": (0.03, 0.08),
    "lifetime_years": (20, 30),
    "crop_price_eur_t": (180., 280.),
    "crop_cost_eur_planted_ha": (800., 1300.),
}


def scenario(table, base, name, value):
    cfg, factor = dict(base), 1.0
    if name == "electricity_price_factor":
        factor = value
    else:
        cfg[name] = value
    return mark_pareto(recompute_economics(table, cfg, factor))


def representatives(table):
    """Per construction: max-LER and max-net Pareto designs (base assumptions)."""
    out = []
    for kind, g in table[table.pareto].groupby("pv_type"):
        for label, col in (("max LER", "LER"), ("max net income", "annual_net_eur_ha")):
            row = g.loc[g[col].idxmax()]
            out.append((kind, label, row.design_id))
    return out


def break_even_module_price(table, cfg):
    """Module price (EUR/kWp) at which annual net income is exactly zero.

    annual_net = E + C - Ccost - kWp*(m + s)*(opex + crf)  ->  solve for m."""
    t = table
    s = structure_rate(t, cfg) + cfg.get("other_capex_eur_kwp", 0.)
    planted = t.planted_area_ha if "planted_area_ha" in t.columns else t.footprint_ha*cfg["planted_fraction"]
    margin = (t.electrical_revenue_eur + t.crop_yield_t_ha.fillna(0.)*planted*cfg["crop_price_eur_t"]
              - planted*cfg["crop_cost_eur_planted_ha"])
    k = cfg["opex_fraction"] + crf(cfg["discount_rate"], cfg["lifetime_years"])
    return margin/(t.installed_kwp*k) - s


def main(run):
    table, meta, folder = load_run(run)
    out = folder/"analysis"
    out.mkdir(exist_ok=True)
    if "config" not in meta:  # recheck folders store no cost config: take it from the search run
        search = f"nsga2_{meta.get('search_sampling', {}).get('mode', 'combined')}"
        print(f"No config in {folder.name}/run_metadata.json; using {search}/run_metadata.json")
        meta["config"] = load_run(search)[1]["config"]
    base = dict(meta["config"])
    err = check_reproduction(table, base)
    print(f"Economics reproduced (max abs error {err:.2e} EUR/ha)")
    table = mark_pareto(table)
    reps = representatives(table)
    base_pareto = set(zip(table[table.pareto].design_id, table[table.pareto].pv_type))

    rows = []
    for name, (lo, hi) in RANGES.items():
        for side, value in (("low", lo), ("high", hi)):
            t = scenario(table, base, name, value)
            par = set(zip(t[t.pareto].design_id, t[t.pareto].pv_type))
            jacc = len(par & base_pareto)/len(par | base_pareto)
            for kind, label, did in reps:
                r = t[(t.design_id == did) & (t.pv_type == kind)].iloc[0]
                rows.append(dict(parameter=name, side=side, value=value, pv_type=kind, design=label,
                                 design_id=did, annual_net_eur_ha=r.annual_net_eur_ha,
                                 payback_years=r.payback_period_years, pareto_jaccard=jacc,
                                 n_pareto=len(par)))
    sens = pd.DataFrame(rows)
    for kind, label, did in reps:
        b = table[(table.design_id == did) & (table.pv_type == kind)].iloc[0]
        m = (sens.pv_type == kind) & (sens.design_id == did)
        sens.loc[m, "base_net_eur_ha"] = b.annual_net_eur_ha
    sens["delta_net_eur_ha"] = sens.annual_net_eur_ha-sens.base_net_eur_ha
    sens.to_csv(out/"sensitivity_oat.csv", index=False)

    stab = sens.groupby(["parameter", "side", "value"], as_index=False)[["pareto_jaccard", "n_pareto"]].first()
    stab.to_csv(out/"pareto_stability.csv", index=False)
    print("\nPareto set stability (Jaccard vs base, 1 = identical):")
    print(stab.round(3).to_string(index=False))

    be = table[["design_id", "pv_type", *decisions(table), "LER", "annual_net_eur_ha", "pareto"]].copy()
    be["break_even_module_eur_kwp"] = break_even_module_price(table, base)
    be.to_csv(out/"break_even_module_price.csv", index=False)
    print("\nBreak-even module price (EUR/kWp) over Pareto designs, base assumptions:")
    print(be[be.pareto].groupby("pv_type").break_even_module_eur_kwp.describe().round(0).to_string())
    print(f"  assumed: semi {base['module_semi_eur_kwp']:.0f}, conventional {base['module_conv_eur_kwp']:.0f}")

    tornado(sens, reps, out)
    print("\nSaved:", out)


def tornado(sens, reps, out):
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    order = (sens.groupby("parameter").delta_net_eur_ha.apply(lambda s: s.abs().max())
             .sort_values().index.tolist())
    y = np.arange(len(order))
    for ax, (kind, label, did) in zip(axes, reps):
        s = sens[(sens.pv_type == kind) & (sens.design_id == did)].set_index(["parameter", "side"])
        lo = [s.loc[(p, "low"), "delta_net_eur_ha"] for p in order]
        hi = [s.loc[(p, "high"), "delta_net_eur_ha"] for p in order]
        ax.barh(y, lo, height=.6, color=COLORS[kind], alpha=.45, label="low value")
        ax.barh(y, hi, height=.6, color=COLORS[kind], label="high value")
        ax.axvline(0, color="#52514e", lw=.8)
        base = s.base_net_eur_ha.iloc[0]
        ax.set_title(f"{LABELS[kind]}: {label}\nbase {base:,.0f} EUR/ha/yr", fontsize=9)
        ax.set_xlabel("Change in annual net income (EUR/ha/yr)")
        ax.grid(axis="x", color="#e6e5e0", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, order, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out/"tornado_net_income.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nsga2_combined")
