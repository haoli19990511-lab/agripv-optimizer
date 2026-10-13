"""Headless Section F: stage-2 re-check of a search's Pareto designs, without Jupyter.

Executes the notebook's model cells (as eval_worker.py does) and its Section D control
panel with RUN_SEARCH forced to False, then calls engine.recheck() exactly like Section F.
Run with the agripv environment from the optimizer folder:
    python analysis/run_recheck.py [pareto_front.csv] [days_per_month] [hours_per_day] [notebook]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results"/"nsga2_combined"/"pareto_front.csv"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
hours = int(sys.argv[3]) if len(sys.argv) > 3 else 4
NOTEBOOK = sys.argv[4] if len(sys.argv) > 4 else "agripv_optimizer_fixed.ipynb"

notebook = json.loads((ROOT/NOTEBOOK).read_text(encoding="utf-8"))
ns = {"__name__": "__main__", "display": print}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    code = "".join(cell["source"])
    if "USER CONTROL PANEL" in code:
        # Settings only: never start a search from here.
        code = code.replace("RUN_SEARCH = True", "RUN_SEARCH = False")
        exec(compile(code, "<control-panel>", "exec"), ns)
        break
    exec(compile(code, "<notebook-cell>", "exec"), ns)
assert ns["RUN_SEARCH"] is False and ns["engine"] is None

import pandas as pd
from sampling_core import SamplingConfig
from weighted_engine import make_engine

engine = make_engine(ns, ns["CROP_CONFIG"], ns["ENGINE_CONFIG"], ns["SAMPLING"], ns["REFERENCE_PV_DESIGN"],
                        ns["PRICE_MODE"], ns["FLAT_PRICE_EUR_KWH"], ns["PRICE_YEARS"],
                        parallel_designs=ns["PARALLEL_DESIGNS"], workers_per_design=ns["WORKERS_PER_DESIGN"],
                        notebook_path=ns["NOTEBOOK_FILE"],
                        constructions=ns.get("CONSTRUCTIONS", ("semi_transparent", "conventional")))
recheck_source = pd.read_csv(source, float_precision="round_trip")
print(f"Re-check source: {source} ({recheck_source.design_id.nunique()} designs); "
      f"sampling combined {days} d/month x {hours} h/day", flush=True)
table = engine.recheck(recheck_source, SamplingConfig(mode="combined", hours_per_day=hours, days_per_month=days))
print(table.loc[table.pareto & table.feasible, ["design_id", "pv_type", "n_rows", "LER", "annual_net_eur_ha", "n_skies"]])
