"""Headless Section D: run the NSGA-II search of one notebook without Jupyter.

Executes the notebook's model cells and its control panel with RUN_SEARCH forced to False,
then runs engine.search() + export() with the panel's settings (POP_SIZE, N_GENERATIONS, SEED,
SAMPLING, PARALLEL_DESIGNS, prices). Redirect stdout to a log file: convergence.py reads it.
    python -u analysis/run_search.py [notebook] > results/search_run.log 2>&1
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
NOTEBOOK = sys.argv[1] if len(sys.argv) > 1 else "agripv_optimizer_fixed.ipynb"

nb = json.loads((ROOT/NOTEBOOK).read_text(encoding="utf-8"))
ns = {"__name__": "__main__", "display": print}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    code = "".join(cell["source"])
    if "USER CONTROL PANEL" in code:
        exec(compile(code.replace("RUN_SEARCH = True", "RUN_SEARCH = False"), "<control-panel>", "exec"), ns)
        break
    exec(compile(code, "<notebook-cell>", "exec"), ns)
assert ns["RUN_SEARCH"] is False and ns["engine"] is None

from weighted_engine import make_engine

engine = make_engine(ns, ns["CROP_CONFIG"], ns["ENGINE_CONFIG"], ns["SAMPLING"], ns["REFERENCE_PV_DESIGN"],
                     ns["PRICE_MODE"], ns["FLAT_PRICE_EUR_KWH"], ns["PRICE_YEARS"],
                     parallel_designs=ns["PARALLEL_DESIGNS"], workers_per_design=ns["WORKERS_PER_DESIGN"],
                     notebook_path=NOTEBOOK, constructions=ns.get("CONSTRUCTIONS", ("semi_transparent", "conventional")))
print(f"Headless search: {NOTEBOOK}, MOUNTING={ns['MOUNTING']}, POP_SIZE={ns['POP_SIZE']}, "
      f"N_GENERATIONS={ns['N_GENERATIONS']}, SEED={ns['SEED']}, costs={ns['ENGINE_CONFIG']}", flush=True)
table = engine.search(ns["POP_SIZE"], ns["N_GENERATIONS"], ns["SEED"], ns["DECISION_BOUNDS"], ns["BASE_DESIGN"])
engine.export(table)
print(table.loc[table.pareto & table.feasible, ["design_id", "pv_type", "LER", "annual_net_eur_ha"]].to_string())
