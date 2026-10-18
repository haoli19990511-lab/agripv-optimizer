"""NSGA-II convergence: hypervolume of the evaluated Pareto set after each generation.

evaluated_candidates.csv has no generation column, so the evaluation order is rebuilt
from the printed log ("NSGA-II <kind> design: {...}" lines and pymoo's n_eval table).
Save the notebook after the search (Ctrl+S) so the cell output is on disk, then:
    python analysis/convergence.py nsga2_combined [log source] [pop_size]
log source = the .ipynb (default: agripv_optimizer_fixed.ipynb) or the log of analysis/run_search.py.
Several runs (seeds) can be compared with the same reference point: --ref LER,net
"""
import ast
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, LABELS, ROOT, decisions, hypervolume_2d, load_run

DESIGN_RE = re.compile(r"NSGA-II (\w+) design: (\{.*?\})")
START_RE = re.compile(r"Starting NSGA-II: (\w+), pop_size=(\d+), n_gen=(\d+)")
# pymoo verbose row: "     3 |       60 | ..." (n_gen | n_eval | ...)
GEN_RE = re.compile(r"^\s*(\d+)\s*\|\s*(\d+)\s*\|", re.M)


def read_log(source):
    source = Path(source)
    if source.suffix == ".ipynb":
        nb = json.loads(source.read_text(encoding="utf-8"))
        parts = []
        for cell in nb["cells"]:
            for o in cell.get("outputs", []):
                parts.append("".join(o.get("text", "")))
        return "\n".join(parts)
    return source.read_text(encoding="utf-8", errors="replace")


def parse(text):
    """Return a list of (kind, generation, design dict) in evaluation order."""
    # Only the most recent search: from the last "Starting NSGA-II: semi_transparent".
    starts = [m for m in START_RE.finditer(text)]
    first = [m for m in starts if m.group(1) == "semi_transparent"]
    if first:
        text = text[first[-1].start():]
    out, kind, gen_bounds = [], None, {}
    blocks = re.split(r"(?=Starting NSGA-II: )", text)
    for block in blocks:
        m = START_RE.match(block)
        if not m:
            continue
        kind, pop = m.group(1), int(m.group(2))
        designs = [ast.literal_eval(d.group(2)) for d in DESIGN_RE.finditer(block) if d.group(1) == kind]
        evals = [int(g.group(2)) for g in GEN_RE.finditer(block)]
        bounds = evals if evals and evals[-1] == len(designs) else list(range(pop, len(designs)+1, pop))
        if len(designs) % pop and not evals:
            bounds.append(len(designs))
        gen = 1
        for i, d in enumerate(designs):
            while gen <= len(bounds) and i >= bounds[gen-1]:
                gen += 1
            out.append((kind, gen, d))
    return out


def attach_generation(table, parsed):
    names = decisions(table)
    key = lambda d: tuple(round(float(d[k]), 6) for k in names)
    first_gen = {}
    for kind, gen, d in parsed:
        first_gen.setdefault((kind, key(d)), gen)
    t = table.copy()
    t["generation"] = [first_gen.get((r.pv_type, key(r)), np.nan) for _, r in t.iterrows()]
    missing = t.generation.isna().sum()
    if missing:
        print(f"WARNING: {missing} rows not found in the log (log from another run?)")
    return t


def hv_curve(t, ref, scale):
    rows = []
    for kind, g in t[t.feasible.astype(bool)].groupby("pv_type"):
        for gen in range(1, int(t.generation.max())+1):
            seen = g[g.generation <= gen]
            pts = (seen[["LER", "annual_net_eur_ha"]].to_numpy(float) - ref[kind])/scale[kind]
            rows.append(dict(pv_type=kind, generation=gen, n_evaluated=len(seen),
                             hypervolume=hypervolume_2d(pts, (0., 0.))))
    return pd.DataFrame(rows)


def main(argv):
    run = argv[0] if argv else "nsga2_combined"
    source = argv[1] if len(argv) > 1 else ROOT/"agripv_optimizer_fixed.ipynb"
    table, _, folder = load_run(run)
    t = attach_generation(table, parse(read_log(source)))
    out = folder/"analysis"
    out.mkdir(exist_ok=True)
    t.to_csv(out/"evaluated_with_generation.csv", index=False)
    # Reference point: worst feasible value minus 5 % of the range; scale = range.
    # Fixed per construction so curves are comparable within (and, by editing, across) runs.
    ref, scale = {}, {}
    for kind, g in t[t.feasible.astype(bool)].groupby("pv_type"):
        v = g[["LER", "annual_net_eur_ha"]].to_numpy(float)
        span = np.maximum(v.max(0)-v.min(0), 1e-9)
        ref[kind], scale[kind] = v.min(0)-.05*span, span
    hv = hv_curve(t, ref, scale)
    hv.to_csv(out/"hypervolume_by_generation.csv", index=False)
    print(hv.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for kind, g in hv.groupby("pv_type"):
        ax.plot(g.generation, g.hypervolume, color=COLORS[kind], lw=2, marker="o", ms=5, label=LABELS[kind])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalised hypervolume")
    ax.grid(color="#e6e5e0", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out/"hypervolume_convergence.png", dpi=200)
    print("Saved:", out)


if __name__ == "__main__":
    main(sys.argv[1:])
