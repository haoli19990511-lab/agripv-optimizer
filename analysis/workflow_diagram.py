"""Workflow diagram of the optimisation framework (Figure 3-1).
    python analysis/workflow_diagram.py   -> results/fig_workflow.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import RESULTS

INK, INK2 = "#0b0b0b", "#52514e"
FILL = {"opt": ("#e3eefb", "#2a78d6"), "rad": ("#fde8df", "#eb6834"), "crop": ("#e2f2e7", "#2f8a4f"),
        "eco": ("#efedf8", "#6b5bb5"), "in": ("#f3f2ee", "#8a8984")}

fig, ax = plt.subplots(figsize=(8.0, 8.4))
ax.set_xlim(0, 112)
ax.set_ylim(0, 118)
ax.axis("off")


def box(x, y, w, h, title, body, kind):
    face, edge = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=face, ec=edge, lw=1.2))
    ax.text(x+w/2, y+h-2.2, title, ha="center", va="top", fontsize=8.3, weight="bold", color=INK)
    if body:
        ax.text(x+w/2, y+h-6.6, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.35)
    return dict(l=(x, y+h/2), r=(x+w, y+h/2), t=(x+w/2, y+h), b=(x+w/2, y), x=x, y=y, w=w, h=h)


def arrow(p, q, text=None, rad=0., dx=1.2, ha="left", color=INK2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0]+q[0])/2+dx, (p[1]+q[1])/2, text, fontsize=6.6, color=INK2, va="center", ha=ha, style="italic")


inp = box(2, 104, 96, 10, "Inputs",
          "TMY weather Aachen (PVGIS)  ·  module data  ·  hourly day-ahead prices 2023–2025  ·  costs and crop prices",
          "in")
opt = box(22, 88, 56, 12, "NSGA-II (pymoo)",
          "population of designs  x = (tilt, pitch, azimuth, clearance)\n(single-axis and vertical systems: own variables, Section 3.3)",
          "opt")
geo = box(22, 75, 56, 9, "Array geometry and sky sampling",
          r"$n_\mathrm{rows}=\lfloor 24\,\mathrm{m}/p\rfloor$  ·  K days × H hours per month", "opt")
rad = box(22, 59, 56, 12, "Radiation model (Liu)",
          "bifacial_radiance + Radiance, Perez sky, middle module traced\nhourly front/rear module irradiance · daily ground irradiation",
          "rad")
pv = box(2, 39, 42, 14, "PV yield",
         r"$E=\sum\,(G_\mathrm{front}+\varphi\,G_\mathrm{rear})\,\eta\,f_T\,f_\mathrm{inv}$" "\n"
         r"$\mathrm{LER_{PV}}=E\,/\,E_\mathrm{ref}$", "rad")
crop = box(56, 39, 42, 14, "Crop model (Li)",
           "WOFOST 7.2 (PCSE), winter wheat, area-\nweighted daily ground irradiation\n"
           r"$\mathrm{LER_{crop}}=f_\mathrm{plant}\,Y_\mathrm{APV}/Y_\mathrm{open}$", "crop")
eco = box(2, 19, 42, 14, "Economics",
          r"$\sum E_h c_h$ + agri-PV bonus + crop revenue" "\n− O&M − crop cost − CAPEX·CRF\n"
          r"→ net income $I_\mathrm{net}$", "eco")
obj = box(56, 19, 42, 14, "Objectives and constraints",
          r"max LER = $\mathrm{LER_{crop}}+\mathrm{LER_{PV}}$,  max $I_\mathrm{net}$" "\n"
          "crop ≥ 66 % · clearance ≥ 2.5 m\ncorridor ≥ 1.5 m", "opt")
fin = box(22, 2, 56, 11, "Re-check and final Pareto front",
          "Pareto designs re-evaluated with denser sampling\n(192 skies: 2 days × 8 hours per month, all mounting types)", "opt")

arrow(inp["b"], opt["t"])
arrow(opt["b"], geo["t"], "candidate design")
arrow(geo["b"], rad["t"])
arrow((rad["x"]+12, rad["y"]), pv["t"], "module irradiance", dx=-1.2, ha="right")
arrow((rad["x"]+rad["w"]-12, rad["y"]), crop["t"], "ground irradiation")
arrow(pv["b"], eco["t"], "E, hourly", dx=1.2)
arrow((crop["x"], crop["y"]+3), (eco["x"]+eco["w"], eco["y"]+eco["h"]-3), None)
ax.text(50, 38.6, "yield", fontsize=6.6, color=INK2, ha="center", style="italic")
arrow(crop["b"], obj["t"], r"$\mathrm{LER_{crop}}$")
arrow(eco["r"], obj["l"], None)
ax.text(50, 23.5, r"$I_\mathrm{net}$, $\mathrm{LER_{PV}}$", fontsize=6.6, color=INK2, ha="center", va="top")
# feedback loop to NSGA-II on the right
c = FILL["opt"][1]
y0, y1, xr = obj["y"]+obj["h"]/2, opt["y"]+opt["h"]/2, 104
ax.plot([obj["x"]+obj["w"]+.6, xr, xr], [y0, y0, y1], color=c, lw=1.1)
ax.add_patch(FancyArrowPatch((xr, y1), (opt["x"]+opt["w"]+.6, y1), arrowstyle="-|>", mutation_scale=10, lw=1.1,
                             color=c, shrinkA=0, shrinkB=1))
ax.text(xr+1.2, (y0+y1)/2, "next generation (6 generations × 20 designs)", fontsize=6.6, color=c, ha="left",
        va="center", style="italic", rotation=90)
arrow(obj["b"], (fin["x"]+fin["w"]-8, fin["y"]+fin["h"]), "after the last generation", dx=3)

fig.savefig(RESULTS/"fig_workflow.png", dpi=300, bbox_inches="tight")
print("Saved", RESULTS/"fig_workflow.png")
