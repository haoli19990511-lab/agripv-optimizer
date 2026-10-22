"""Joint Pareto front over the three mounting types (conventional modules, agri-PV bonus).

    python analysis/joint_front.py
Reads results/analysis_mountings/mountings_pareto.csv (written by compare_mountings.py), keeps the designs that no
design of any mounting type dominates in LER and annual net income, and writes to results/analysis_mountings/:
  joint_front.csv        designs of the joint front, sorted by LER
  fig_joint_front.png    the three fronts and the joint front
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS, pareto_mask

OUT = RESULTS/"analysis_mountings"
COL = {"Fixed tilt": "#2a78d6", "Single-axis": "#6b5bb5", "Vertical": "#2f8a4f"}  # as in Figure 4-4
MARK = {"Fixed tilt": "o", "Single-axis": "D", "Vertical": "^"}
INK, MUTED, GRID = "#1E2B26", "#52514e", "#e6e5e0"

fronts = pd.read_csv(OUT/"mountings_pareto.csv", float_precision="round_trip").reset_index(drop=True)
joint = fronts[pareto_mask(fronts[["LER", "annual_net_eur_ha"]].values)].sort_values("LER")
joint.to_csv(OUT/"joint_front.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=250)
for name, g in fronts.groupby("mounting_type", sort=False):
    g = g.sort_values("LER")
    ax.plot(g.LER, g.annual_net_eur_ha/1000, color=COL[name], lw=1.4, alpha=.45, marker=MARK[name], ms=5,
            mec="white", mew=.6, label=f"{name} front (n={len(g)})", zorder=2)
ax.plot(joint.LER, joint.annual_net_eur_ha/1000, color=INK, lw=2.2, zorder=3, label=f"Joint front (n={len(joint)})")
for name, g in joint.groupby("mounting_type"):
    ax.scatter(g.LER, g.annual_net_eur_ha/1000, s=46, color=COL[name], marker=MARK[name], edgecolor="white",
               linewidth=.8, zorder=4)
labels = [(1.141, 2.58, "Vertical, 2 rows\n+2,578", (1.135, -8.5)),
          (1.210, 0.95, "Fixed tilt, 2 rows\n+954", (1.215, 3.4)),
          (1.269, 0.50, "Vertical, 4 rows", (1.272, 2.6)),
          (1.324, 0.14, "Fixed tilt, 3 rows", (1.345, 3.4)),
          (1.520, -5.02, "Fixed tilt, 5 rows\n−5,024", (1.43, -9.5)),
          (1.570, -27.83, "Highest LER 1.57\n−27,832", (1.46, -25))]
for x, y, text, xy in labels:
    ax.annotate(text, (x, y), xytext=xy, fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7))
ax.axhline(0, color=MUTED, lw=.8, zorder=1)
ax.set_xlabel("Land equivalent ratio LER (–)", fontsize=10, color=INK)
ax.set_ylabel("Annual net income (k EUR/(ha·a))", fontsize=10, color=INK)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(color=GRID, lw=.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_ylim(-32, 5.5)
fig.tight_layout()
fig.savefig(OUT/"fig_joint_front.png", dpi=250)
print(joint[["mounting_type", "n_rows", "pitch", "LER", "annual_net_eur_ha"]].round(3).to_string(index=False))
print("Saved:", OUT/"fig_joint_front.png")
