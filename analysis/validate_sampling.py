"""Validate the representative sampling against the full hourly year (fixed tilt).

Takes 3 Pareto designs of a result table (max net income, max LER and the one in between),
conventional modules only, and evaluates each with the search sampling (24 skies), the re-check
sampling (144), 8 hours per day (96) and the FULL year (every lit hour), including the crop model.
Cheap samplings run first; everything is cached, so an interrupted run resumes.
Run with the agripv environment from the optimizer folder:
    python analysis/validate_sampling.py [results/<run>/pareto_front.csv] [parallel]
Output: results/sampling_validation/sampling_validation.csv and errors_vs_full.csv
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results"/"recheck_combined_144skies"/"pareto_front.csv"
parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NOTEBOOK = "agripv_optimizer_fixed.ipynb"

nb = json.loads((ROOT/NOTEBOOK).read_text(encoding="utf-8"))
ns = {"__name__": "__main__", "display": print}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    code = "".join(cell["source"])
    if "USER CONTROL PANEL" in code:
        exec(compile(code.replace("RUN_SEARCH = True", "RUN_SEARCH = False"), "<panel>", "exec"), ns)
        break
    exec(compile(code, "<cell>", "exec"), ns)
assert ns["MOUNTING"] == "fixed"

import pandas as pd
from sampling_core import SamplingConfig
from weighted_engine import make_engine

SAMPLINGS = {  # cheapest first; "full" = every lit hour of the year (the reference)
    "search 1d x 2h": SamplingConfig(mode="combined", days_per_month=1, hours_per_day=2),
    "recheck 3d x 4h": SamplingConfig(mode="combined", days_per_month=3, hours_per_day=4),
    "1d x 8h": SamplingConfig(mode="combined", days_per_month=1, hours_per_day=8),
    "full year": SamplingConfig(mode="full"),
}

pareto = pd.read_csv(source, float_precision="round_trip")
conv = pareto[(pareto.pv_type == "conventional") & pareto.feasible].sort_values("LER").reset_index(drop=True)
if len(conv) < 3:
    raise ValueError(f"Need at least 3 conventional Pareto designs in {source}")
picks = {"max net income": conv.annual_net_eur_ha.idxmax(), "max LER": conv.LER.idxmax()}
middle = [i for i in conv.index if i not in picks.values()]
picks["in between"] = middle[len(middle)//2]

out = ROOT/"results"/"sampling_validation"
out.mkdir(parents=True, exist_ok=True)
rows = []
for label, sampling in SAMPLINGS.items():
    engine = make_engine(ns, ns["CROP_CONFIG"], ns["ENGINE_CONFIG"], sampling, ns["REFERENCE_PV_DESIGN"],
                         ns["PRICE_MODE"], ns["FLAT_PRICE_EUR_KWH"], ns["PRICE_YEARS"],
                         parallel_designs=parallel, workers_per_design=None, notebook_path=NOTEBOOK,
                         constructions=("conventional",))
    designs = {name: engine._design_from_row(conv.loc[i]) for name, i in picks.items()}
    print(f"=== {label}: {engine.n_skies(sampling)} skies per design", flush=True)
    engine.precompute(list(designs.values()), sampling)
    for name, design in designs.items():
        row = next(r for r in engine.evaluate(design, sampling) if r["pv_type"] == "conventional")
        row.update(sampling=label, n_skies=engine.n_skies(sampling), design=name,
                   kwh_per_kwp=row["electrical_kwh"]/row["installed_kwp"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"sampling_validation.csv", index=False)  # saved after every sampling

table = pd.DataFrame(rows)
metrics = ["kwh_per_kwp", "LER_pv", "LER_crop", "LER", "annual_net_eur_ha"]
full = table[table.sampling == "full year"].set_index("design")[metrics]
errors = []
for label, g in table.groupby("sampling", sort=False):
    g = g.set_index("design")[metrics]
    e = (100*(g/full - 1)).add_suffix("_error_%")
    e["sampling"] = label
    for m in ("LER", "annual_net_eur_ha"):
        e[f"rank_{m}"] = g[m].rank(ascending=False).astype(int)
    errors.append(e.reset_index())
errors = pd.concat(errors, ignore_index=True)
errors.to_csv(out/"errors_vs_full.csv", index=False)
print(errors.round(2).to_string(index=False))
print("Saved:", out)
