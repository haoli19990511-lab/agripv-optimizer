"""Section 4.1 material from the final fixed-tilt re-check (no physics needed).
    python analysis/results_fixed.py recheck_combined_192skies
Writes to results/<run>/analysis/:
  fig_fixed_pareto_bonus.png   Pareto fronts with the agri-PV bonus, conventional front also without it
  fig_fixed_pareto_geometry.png  decision variables along the conventional front
  fixed_pareto_table.csv       conventional and semi-transparent Pareto designs, net income with/without bonus
  monthly_capture_price.csv    monthly PV-weighted vs plain price for two conventional Pareto designs
  fixed_summary.json           numbers quoted in the text
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, GREY, LABELS, ROOT, check_reproduction, load_run, mark_pareto, recompute_economics

sys.path.insert(0, str(ROOT))
from weighted_engine import load_price_profile  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
MODULE_Y = 2.278  # conventional module length across the row (m), for the clear corridor note only


def style(ax):
    ax.grid(color=GRID, lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def with_no_bonus(t, config):
    t = t.copy()
    t["net_no_bonus_eur_ha"] = recompute_economics(t, dict(config, agripv_bonus_eur_kwh=0.)).annual_net_eur_ha
    t["kwh_per_kwp"] = t.electrical_kwh/t.installed_kwp
    t["capex_eur_kwp"] = t.capex_total_eur/t.installed_kwp
    return t


def fig_fronts(t, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(t.LER, t.annual_net_eur_ha/1000, s=12, color=GREY, alpha=.5, lw=0, label="Re-checked designs")
    conv = t[(t.pv_type == "conventional") & t.pareto].sort_values("LER")
    nob = t[(t.pv_type == "conventional") & t.pareto_no_bonus].sort_values("LER")
    semi = t[(t.pv_type == "semi_transparent") & t.pareto].sort_values("LER")
    ax.plot(conv.LER, conv.annual_net_eur_ha/1000, color=COLORS["conventional"], lw=2, marker="o", ms=6,
            mec="white", mew=1.2, label=f"Conventional, with bonus (n={len(conv)})")
    ax.plot(nob.LER, nob.net_no_bonus_eur_ha/1000, color=COLORS["conventional"], lw=1.6, ls="--", marker="o", ms=5,
            mfc="white", mew=1.2, label=f"Conventional, without bonus (n={len(nob)})")
    ax.plot(semi.LER, semi.annual_net_eur_ha/1000, color=COLORS["semi_transparent"], lw=2, marker="s", ms=5,
            mec="white", mew=1.2, label=f"Semi-transparent, with bonus (n={len(semi)})")
    ax.axhline(0, color=INK2, lw=.8)
    for rows, g in conv.groupby("n_rows"):  # row count of the front designs, at the top edge
        ax.text(g.LER.mean(), 1.0, f"{rows} rows", transform=ax.get_xaxis_transform(), fontsize=8,
                color=INK2, ha="center", va="bottom")
    ax.set_xlabel("Land equivalent ratio LER (–)")
    ax.set_ylabel("Annual net income (k EUR/(ha·a))")
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out/"fig_fixed_pareto_bonus.png", dpi=250)
    plt.close(fig)


def fig_geometry(t, out):
    conv = t[(t.pv_type == "conventional") & t.pareto].sort_values("LER")
    spec = [("n_rows", "Rows in the field (–)"), ("pitch", "Pitch (m)"), ("tilt", "Tilt (°)"),
            ("clearance_height", "Clearance (m)")]
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6), sharex=True)
    for ax, (col, label) in zip(axes, spec):
        ax.scatter(conv.LER, conv[col], s=28, color=COLORS["conventional"], edgecolors="white", linewidths=1, zorder=3)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("LER (–)", fontsize=8)
        style(ax)
    axes[0].set_yticks([2, 3, 4, 5])
    axes[3].axhline(2.5, color=INK2, lw=.8, ls="--")
    axes[2].axhline(50, color=INK2, lw=.8, ls="--")
    axes[1].axhline(4, color=INK2, lw=.8, ls="--")
    fig.tight_layout()
    fig.savefig(out/"fig_fixed_pareto_geometry.png", dpi=250)
    plt.close(fig)


def monthly_capture(t, meta, out):
    """Monthly PV-weighted price from the cached hourly yield of two conventional Pareto designs."""
    s = meta["sampling"]
    conv = t[(t.pv_type == "conventional") & t.pareto]
    picks = {"highest income": conv.loc[conv.annual_net_eur_ha.idxmax()], "highest LER": conv.loc[conv.LER.idxmax()]}
    manifests = []
    for m in (ROOT/"results"/"cache_v5").glob("*/manifest.json"):
        j = json.loads(m.read_text())
        if j.get("sampling", {}).get("days_per_month") == s["days_per_month"] and \
                j["sampling"].get("hours_per_day") == s["hours_per_day"] and "tilt" in j.get("design", {}):
            manifests.append((m.parent, j["design"]))
    rows = []
    prices = None
    for name, d in picks.items():
        hits = [f for f, dd in manifests if all(abs(dd[k]-d[k]) < 1e-9 for k in ("tilt", "pitch", "azimuth", "clearance_height"))]
        if not hits:
            raise FileNotFoundError(f"No cached hourly yield for {name}")
        hourly = pd.read_csv(hits[0]/"hourly.csv.gz")
        hours = pd.DatetimeIndex(pd.to_datetime(hourly.hour_start))
        if prices is None:
            prices = load_price_profile(hours, ROOT/"prices", "csv", None, meta["price_assumptions"]["years"])
        kwh = sum(hourly[f"conventional_array_{f}_effective_incident_energy_Wh_hour"].to_numpy(float) for f in ("front", "rear"))
        frame = pd.DataFrame({"month": hours.month, "kwh": kwh, "rev": kwh*prices.to_numpy(), "price": prices.to_numpy()})
        g = frame.groupby("month")
        res = pd.DataFrame({"design": name, "energy_share": g.kwh.sum()/frame.kwh.sum(),
                            "capture_price_ct_kwh": 100*g.rev.sum()/g.kwh.sum(),
                            "average_price_ct_kwh": 100*g.price.mean()})
        res.loc[13] = [name, 1., 100*frame.rev.sum()/frame.kwh.sum(), 100*frame.price.mean()]
        res["negative_price_energy_share"] = np.nan
        res.loc[13, "negative_price_energy_share"] = frame.kwh[frame.price < 0].sum()/frame.kwh.sum()
        rows.append(res.rename_axis("month").reset_index())
    table = pd.concat(rows, ignore_index=True)
    table.loc[table.month == 13, "month"] = "year"
    table.round(4).to_csv(out/"monthly_capture_price.csv", index=False)
    return table


def main(run):
    table, meta, folder = load_run(run)
    out = folder/"analysis"
    out.mkdir(exist_ok=True)
    check_reproduction(table, meta["config"])
    t = mark_pareto(with_no_bonus(table, meta["config"]))
    t["pareto_no_bonus"] = mark_pareto(t, ("LER", "net_no_bonus_eur_ha")).pareto
    fig_fronts(t, out)
    fig_geometry(t, out)
    cols = ["pv_type", "design_id", "n_rows", "tilt", "pitch", "azimuth", "clearance_height", "LER_crop", "LER_pv", "LER",
            "kwh_per_kwp", "capex_eur_kwp", "annual_net_eur_ha", "net_no_bonus_eur_ha", "payback_period_years"]
    pareto = t[t.pareto][cols].sort_values(["pv_type", "LER"])
    pareto.round(3).to_csv(out/"fixed_pareto_table.csv", index=False)
    monthly = monthly_capture(t, meta, out)
    conv = t[t.pv_type == "conventional"]
    cp = t[(t.pv_type == "conventional") & t.pareto]
    sp = t[(t.pv_type == "semi_transparent") & t.pareto]
    summary = dict(
        n_designs=int(t.design_id.nunique()), n_pareto_conv=len(cp), n_pareto_semi=len(sp),
        n_pareto_conv_no_bonus=int(t[(t.pv_type == "conventional") & t.pareto_no_bonus].shape[0]),
        conv_LER_range=[cp.LER.min(), cp.LER.max()], conv_net_range=[cp.annual_net_eur_ha.min(), cp.annual_net_eur_ha.max()],
        semi_LER_range=[sp.LER.min(), sp.LER.max()], semi_net_range=[sp.annual_net_eur_ha.min(), sp.annual_net_eur_ha.max()],
        conv_positive_with_bonus=int((conv.annual_net_eur_ha > 0).sum()), conv_positive_no_bonus=int((conv.net_no_bonus_eur_ha > 0).sum()),
        n_conv=len(conv),
        bonus_eur_ha_range=[float((conv.annual_net_eur_ha-conv.net_no_bonus_eur_ha).min()),
                            float((conv.annual_net_eur_ha-conv.net_no_bonus_eur_ha).max())],
        capture_price_ct=float(100*conv.capture_price_eur_kwh.median()),
        semi_dominated=bool(all(((cp.LER.values >= s.LER) & (cp.annual_net_eur_ha.values >= s.annual_net_eur_ha)).any()
                                for _, s in sp.iterrows())),
        kwh_per_kwp_conv=[conv.kwh_per_kwp.min(), conv.kwh_per_kwp.max()],
        kwh_per_kwp_semi=[t[t.pv_type == "semi_transparent"].kwh_per_kwp.min(), t[t.pv_type == "semi_transparent"].kwh_per_kwp.max()],
    )
    (out/"fixed_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(pareto.round(3).to_string(index=False))
    print(monthly.round(3).to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recheck_combined_192skies")
