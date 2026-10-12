"""Worker process: runs the expensive Radiance physics for ONE design (or the
reference design) and leaves the result in the shared disk cache.

Started by WeightedEngine.precompute(); not meant to be run by hand.
It re-executes the notebook's setup cells (everything before the Section D
control panel) so it uses exactly the same model code as the notebook, then
checks that its cache signature equals the parent's. A mismatch usually means
the notebook has unsaved edits: the worker exits with code 3 instead of writing
results under a different cache key.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import sys


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    os.chdir(spec["cwd"])
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    notebook = json.loads(Path(spec["notebook"]).read_text(encoding="utf-8"))
    ns = {"__name__": "__main__", "display": print}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "USER CONTROL PANEL" in source:
            break  # Section D onwards is the run control, not model code.
        exec(compile(source, "<notebook-cell>", "exec"), ns)
    ns["SETTINGS"] = replace(ns["SETTINGS"], n_workers=int(spec["workers"]))

    import bridge
    from sampling_core import SamplingConfig, fingerprint
    from weighted_engine import EngineConfig, make_engine

    sampling = SamplingConfig(**spec["sampling"])
    engine = make_engine(ns, bridge.CropConfig(**spec["crop"]), EngineConfig(**spec["config"]), sampling,
                            ns["PVDesign"](**spec["reference"]), spec["price_mode"], spec["flat_price"],
                            spec["price_years"],
                            constructions=spec.get("constructions", ["semi_transparent", "conventional"]))
    if fingerprint(engine.physics_signature) != spec["signature_sha"]:
        print("ERROR: worker cache signature differs from the notebook's. Save the notebook (Ctrl+S) "
              "after any edit to model cells, and restart the kernel if packages changed.", flush=True)
        return 3
    if spec["kind"] == "reference":
        value = engine.reference_yield(sampling)
        print(f"reference PV yield {value:.1f} kWh/ha cached", flush=True)
    else:
        from weighted_engine import DESIGN_CLASSES, build_design
        design = build_design(ns[DESIGN_CLASSES[ns.get("MOUNTING", "fixed")]], spec["design"])
        _, _, folder = engine._simulation(design, sampling, crop=True)
        print(f"physics cached: {folder}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
