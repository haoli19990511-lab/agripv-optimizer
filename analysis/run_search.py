"""Headless Section D: run the NSGA-II search of one notebook without Jupyter.

Executes the notebook's model cells and its control panel with RUN_SEARCH forced to False,
then runs engine.search() + export() with the panel's settings (POP_SIZE, N_GENERATIONS, SEED,
SAMPLING, PARALLEL_DESIGNS, prices). Redirect stdout to a log file: convergence.py reads it.
    python -u analysis/run_search.py [notebook] [--seed N] [--tag NAME] [--conventional-only] [--crop KEY]
--seed overrides SEED of the control panel; --tag moves the exported folder to <folder>_<tag>
(so repeated runs do not overwrite each other); --conventional-only optimises conventional modules only;
--crop selects a preset of crops.CROP_PRESETS instead of the CROP of the control panel.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import argparse
args = argparse.ArgumentParser()
args.add_argument("notebook", nargs="?", default="agripv_optimizer_fixed.ipynb")
args.add_argument("--seed", type=int)
args.add_argument("--tag")
args.add_argument("--conventional-only", action="store_true")
args.add_argument("--crop")
args = args.parse_args()
NOTEBOOK = args.notebook

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

import shutil
from weighted_engine import make_engine
if args.seed is not None:
    ns["SEED"] = args.seed
if args.conventional_only:
    ns["CONSTRUCTIONS"] = ("conventional",)
if args.crop:
    import crops
    ns["CROP_CONFIG"], ns["ENGINE_CONFIG"] = crops.crop_setup(args.crop, ns["ENGINE_CONFIG"])

engine = make_engine(ns, ns["CROP_CONFIG"], ns["ENGINE_CONFIG"], ns["SAMPLING"], ns["REFERENCE_PV_DESIGN"],
                     ns["PRICE_MODE"], ns["FLAT_PRICE_EUR_KWH"], ns["PRICE_YEARS"],
                     parallel_designs=ns["PARALLEL_DESIGNS"], workers_per_design=ns["WORKERS_PER_DESIGN"],
                     notebook_path=NOTEBOOK, constructions=ns.get("CONSTRUCTIONS", ("semi_transparent", "conventional")))
print(f"Headless search: {NOTEBOOK}, MOUNTING={ns['MOUNTING']}, POP_SIZE={ns['POP_SIZE']}, "
      f"N_GENERATIONS={ns['N_GENERATIONS']}, SEED={ns['SEED']}, crop={ns['CROP_CONFIG'].crop_name}/"
      f"{ns['CROP_CONFIG'].variety}, costs={ns['ENGINE_CONFIG']}", flush=True)
table = engine.search(ns["POP_SIZE"], ns["N_GENERATIONS"], ns["SEED"], ns["DECISION_BOUNDS"], ns["BASE_DESIGN"])
folder = engine._results_folder(f"nsga2_{engine.sampling.mode}")
if args.tag:
    # export() always writes <folder>; park an existing result there so it is never overwritten.
    target = folder.with_name(f"{folder.name}_{args.tag}")
    parked = folder.with_name(f"{folder.name}__parked_by_{args.tag}")
    if target.exists() or parked.exists():
        raise FileExistsError(f"{target} or {parked} exists; choose another --tag")
    if folder.exists():
        shutil.move(str(folder), str(parked))
    try:
        engine.export(table)
        shutil.move(str(folder), str(target))
    finally:
        if parked.exists():
            shutil.move(str(parked), str(folder))
    print("Saved results to", target, flush=True)
else:
    engine.export(table)
print(table.loc[table.pareto & table.feasible, ["design_id", "pv_type", "LER", "annual_net_eur_ha"]].to_string())
