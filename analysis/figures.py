"""Thesis figures from one results folder (no physics needed).
    python analysis/figures.py nsga2_combined
Writes to results/<run>/analysis/:
  fig_pareto_front.png        LER vs annual net income, all evaluated + Pareto per construction
  fig_ler_components.png      LER_crop vs LER_pv (the land-sharing trade-off)
  fig_decisions_vs_kpi.png    Spearman correlation decision variable -> KPI, per construction
  fig_pareto_decisions.png    decision values of Pareto designs against the search bounds
  pareto_summary.csv          compact Pareto table for the thesis
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, GREY, KPIS, LABELS, decisions, load_run, mark_pareto

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
UNITS = {"tilt": "Tilt (°)", "pitch": "Pitch (m)", "azimuth": "Azimuth (°)",
         "clearance_height": "Clearance (m)", "axis_azimuth": "Axis azimuth (°)",
         "hub_height": "Hub height (m)", "limit_angle": "Rotation limit (°)", "LER": "LER", "LER_crop": "LER crop",
         "LER_pv": "LER PV", "annual_net_eur_ha": "Net income"}


def style(ax):
    ax.grid(color=GRID, lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def pareto_front(t, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in ("conventional", "semi_transparent"):
        g = t[t.pv_type == kind]
        p = g[g.pareto].sort_values("LER")
        ax.scatter(g.LER, g.annual_net_eur_ha/1000, s=14, color=COLORS[kind], alpha=.25, lw=0)
        ax.plot(p.LER, p.annual_net_eur_ha/1000, color=COLORS[kind], lw=2, marker="o", ms=6,
                mec="white", mew=1.2, label=f"{LABELS[kind]} (Pareto, n={len(p)})")
    inf = t[~t.feasible.astype(bool)]
    if len(inf):
        ax.scatter(inf.LER, inf.annual_net_eur_ha/1000, s=14, marker="x", color=GREY, label="infeasible")
    ax.axhline(0, color=INK2, lw=.8)
    ax.set_xlabel("Land equivalent ratio (LER)")
    ax.set_ylabel("Annual net income (k EUR/ha/yr)")
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"fig_pareto_front.png", dpi=200)
    plt.close(fig)


def ler_components(t, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    for kind in ("conventional", "semi_transparent"):
        g = t[t.pv_type == kind]
        ax.scatter(g.LER_pv, g.LER_crop, s=16, color=COLORS[kind], alpha=.6, lw=0, label=LABELS[kind])
    x = np.linspace(0, max(1, t.LER_pv.max()), 50)
    for level in (1.2, 1.4, 1.6):
        ax.plot(x, level-x, color=GRID, lw=1, zorder=0)
        ax.text(x[-1], level-x[-1], f" LER={level}", fontsize=7, color=INK2, va="center")
    ax.set_xlim(0, t.LER_pv.max()*1.15)
    ax.set_ylim(min(.6, t.LER_crop.min()-.02), 1.01)
    ax.set_xlabel("LER PV")
    ax.set_ylabel("LER crop")
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"fig_ler_components.png", dpi=200)
    plt.close(fig)


def correlations(t, out):
    DECISIONS = decisions(t)
    kinds = ["conventional", "semi_transparent"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    rows = []
    for ax, kind in zip(axes, kinds):
        g = t[(t.pv_type == kind) & t.feasible.astype(bool)]
        c = g[DECISIONS+KPIS].corr(method="spearman").loc[DECISIONS, KPIS]
        for d in DECISIONS:
            for k in KPIS:
                rows.append(dict(pv_type=kind, decision=d, kpi=k, spearman=c.loc[d, k]))
        im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)  # diverging, grey-ish midpoint
        for i in range(len(DECISIONS)):
            for j in range(len(KPIS)):
                v = c.iloc[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > .6 else INK)
        ax.set_xticks(range(len(KPIS)), [UNITS[k] for k in KPIS], fontsize=8, rotation=20)
        ax.set_yticks(range(len(DECISIONS)), [UNITS[d] for d in DECISIONS], fontsize=8)
        ax.set_title(f"{LABELS[kind]} (n={len(g)})", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.8, label="Spearman ρ")
    fig.savefig(out/"fig_decisions_vs_kpi.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"spearman_decisions_kpi.csv", index=False)


def pareto_decisions(t, bounds, out):
    DECISIONS = decisions(t)
    fig, axes = plt.subplots(1, len(DECISIONS), figsize=(10, 3))
    for ax, d in zip(axes, DECISIONS):
        for i, kind in enumerate(("conventional", "semi_transparent")):
            v = t[(t.pv_type == kind) & t.pareto][d]
            ax.scatter(np.full(len(v), i)+np.random.default_rng(0).uniform(-.12, .12, len(v)), v,
                       s=22, color=COLORS[kind], edgecolors="white", linewidths=.8)
        if d in bounds:
            for b in bounds[d]:
                ax.axhline(b, color=INK2, lw=1, ls="--")
        ax.set_xticks([0, 1], ["Conv.", "Semi"], fontsize=8)
        ax.set_xlim(-.5, 1.5)
        ax.set_title(UNITS[d], fontsize=9)
        style(ax)
    fig.suptitle("Pareto designs vs search bounds (dashed): points on a dashed line = optimum at bound",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out/"fig_pareto_decisions.png", dpi=200)
    plt.close(fig)


def main(run, bounds=None):
    table, meta, folder = load_run(run)
    out = folder/"analysis"
    out.mkdir(exist_ok=True)
    t = mark_pareto(table)
    if bounds is None:  # notebook defaults (v7 Section C)
        names = decisions(t)
        bounds = ({"pitch": (4., 12.), "axis_azimuth": (120., 240.), "hub_height": (3., 5.5), "limit_angle": (30., 60.)}
                  if "axis_azimuth" in names else
                  {"pitch": (4., 12.), "azimuth": (0., 180.), "clearance_height": (.3, 1.5)}
                  if "tilt" not in names else
                  {"tilt": (10., 50.), "pitch": (4., 12.), "azimuth": (120., 240.), "clearance_height": (2.5, 5.)})
    pareto_front(t, out)
    ler_components(t, out)
    correlations(t, out)
    pareto_decisions(t, bounds, out)
    cols = ["pv_type", "design_id"]+decisions(t)+["n_rows", "LER_crop", "LER_pv", "LER", "annual_net_eur_ha",
                                               "payback_period_years", "installed_kwp"]
    t[t.pareto][cols].sort_values(["pv_type", "LER"]).round(3).to_csv(out/"pareto_summary.csv", index=False)
    print("Saved figures:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nsga2_combined")
