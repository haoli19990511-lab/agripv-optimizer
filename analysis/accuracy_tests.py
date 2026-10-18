"""Accuracy tests of the model simplifications (thesis Section 4.7: Table 11, Table 12, single-module tracing).

These tests change settings that enter the ray tracing, so they need NEW RAY TRACING (unlike reproduce_thesis.py).
They run in this process (not in parallel workers, which would reject the changed settings).
    python analysis/accuracy_tests.py grid    [--dry-run]                 Table 11: ground grid 12 / 48 / 96 points
    python analysis/accuracy_tests.py hours   [--dry-run]                 Table 12: 1 / 2 / 4 / 8 sampled hours per day
    python analysis/accuracy_tests.py trace   [--dry-run] [--both]        middle module versus all modules traced
--dry-run lists the configurations and whether their ray tracing is already cached. --both adds the
semi-transparent module to the tracing test (5-8 times slower). Approximate computing time on the thesis PC:
grid 1-2 h, hours about 1 h, trace (conventional modules) 5-8 h.
Output: results/accuracy_tests/<test>.csv, compared with the values in the thesis.
"""
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from common import RESULTS, ROOT

os.chdir(ROOT)  # the notebooks resolve weather/ and results/ from the working directory
sys.path.insert(0, str(ROOT))
from sampling_core import SamplingConfig, resolve_array  # noqa: E402
from weighted_engine import make_engine  # noqa: E402

OUT = RESULTS/"accuracy_tests"
SEARCH_SAMPLING = SamplingConfig(mode="combined", hours_per_day=2, days_per_month=1)  # 24 skies
GAPS = dict(xcellgap=.0004, ycellgap=.00235, xgap_conventional=.2)

# Table 11: five-row design of the search at three pitches (tilt, azimuth and clearance as found by NSGA-II)
GRID_DESIGN = dict(tilt=45.78863448784694, azimuth=191.23551823684704, clearance_height=2.8513395241068764)
GRID_RUNS = [(4.05, 12), (4.05, 48), (4.05, 96), (4.30, 48), (4.80, 48)]           # (pitch, ground points in depth)
GRID_THESIS = {(4.05, 12): .730, (4.05, 48): .827, (4.05, 96): .824, (4.30, 48): .825, (4.80, 48): .836}

# Single-module tracing: four Pareto designs of the search with two to five rows
TRACE_DESIGNS = {
    "84d1851392429941": dict(tilt=31.98374750692238, pitch=4.220472905944547, azimuth=210.4215730409768,
                             clearance_height=3.8453582830481956),
    "f5695141cc26a27b": dict(tilt=22.47325808041942, pitch=7.386611591780605, azimuth=219.324311258453,
                             clearance_height=3.522997840922903),
    "c57d698a7af2ac74": dict(tilt=30.47286498801027, pitch=11.603709570607482, azimuth=137.29915352635604,
                             clearance_height=4.87162361784311),
    "de87fc9ca7378289": dict(tilt=23.189268659963687, pitch=10.307429627427235, azimuth=156.3833795149974,
                             clearance_height=3.6337447237016285),
}
TRACE_THESIS = "kWh −1.2 % to +0.2 %; crop < 0.05 %; reference −3.6 %; LER_PV +2.5 to +3.9 %; LER +0.7 to +1.0 %"

# Table 12: fixed tilt 35° and a single-axis tracker with the same pitch, conventional modules, 1 day per month
HOURS = (1, 2, 4, 8)
HOURS_THESIS = {1: (1517, 1287), 2: (1440, 1265), 4: (1428, 1407), 8: (1389, 1450)}


def notebook(name, **settings):
    """Model cells and control panel of a notebook (no search), with changed SETTINGS fields."""
    nb = json.loads((ROOT/name).read_text(encoding="utf-8"))
    ns = {"__name__": "__main__", "display": print}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            if "USER CONTROL PANEL" in code:
                exec(compile(code.replace("RUN_SEARCH = True", "RUN_SEARCH = False"), "<panel>", "exec"), ns)
                break
            exec(compile(code, "<cell>", "exec"), ns)
    if settings:
        ns["SETTINGS"] = replace(ns["SETTINGS"], **settings)
    return ns


def engine_for(ns, constructions):
    with contextlib.redirect_stdout(io.StringIO()):
        return make_engine(ns, ns["CROP_CONFIG"], ns["ENGINE_CONFIG"], SEARCH_SAMPLING, ns["REFERENCE_PV_DESIGN"],
                           ns["PRICE_MODE"], ns["FLAT_PRICE_EUR_KWH"], ns["PRICE_YEARS"], parallel_designs=1,
                           workers_per_design=None, notebook_path=ns["NOTEBOOK_FILE"], constructions=constructions)


def evaluate(engine, design, label, dry):
    cached = engine._physics_cached(design, SEARCH_SAMPLING)
    print(f"  {label}: ray tracing {'cached' if cached else 'NEW'}", flush=True)
    if dry:
        return []
    t0 = time.time()
    rows = engine.evaluate(design, SEARCH_SAMPLING)
    print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
    return rows


def test_grid(dry):
    rows = []
    for pitch, points in GRID_RUNS:
        ns = notebook("agripv_optimizer_fixed.ipynb")
        ns["scenario"] = replace(ns["scenario"], ground_grid_y=points)
        engine = engine_for(ns, ("conventional",))
        design = ns["PVDesign"](pitch=pitch, **GRID_DESIGN, **GAPS)
        for r in evaluate(engine, design, f"pitch {pitch} m, {points} ground points", dry):
            rows.append(dict(pitch=pitch, ground_points=points, spacing_m=24/points, LER_crop=r["LER_crop"],
                             LER_pv=r["LER_pv"], LER=r["LER"], thesis_LER_crop=GRID_THESIS[(pitch, points)]))
    return pd.DataFrame(rows)


def test_trace(dry, both):
    constructions = ("semi_transparent", "conventional") if both else ("conventional",)
    rows = []
    for mode in ("center", "all"):
        ns = notebook("agripv_optimizer_fixed.ipynb", pv_trace_mode=mode)
        engine = engine_for(ns, constructions)
        for design_id, values in TRACE_DESIGNS.items():
            design = ns["PVDesign"](**values, **GAPS)
            for r in evaluate(engine, design, f"{mode}: {design_id}", dry):
                rows.append(dict(mode=mode, design_id=design_id, pv_type=r["pv_type"], n_rows=r["n_rows"],
                                 electrical_kwh=r["electrical_kwh"], crop_yield_t_ha=r["crop_yield_t_ha"],
                                 reference_kwh_ha=r["pv_reference_kwh_ha"], LER_pv=r["LER_pv"], LER=r["LER"]))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    w = t.pivot_table(index=["design_id", "pv_type", "n_rows"], columns="mode",
                      values=["electrical_kwh", "crop_yield_t_ha", "reference_kwh_ha", "LER_pv", "LER"])
    change = pd.DataFrame({f"{m}_change_%": 100*(w[(m, "center")]/w[(m, "all")]-1) for m in w.columns.levels[0]})
    return change.reset_index()


def test_hours(dry):
    rows = []
    for mounting, nb_name in (("fixed_35deg", "agripv_optimizer_fixed.ipynb"),
                              ("single_axis", "agripv_optimizer_single_axis.ipynb")):
        ns = notebook(nb_name)
        design = (ns["PVDesign"](tilt=35., pitch=7., azimuth=180., clearance_height=3., **GAPS) if mounting == "fixed_35deg"
                  else ns["TrackingDesign"](pitch=7., axis_azimuth=180., hub_height=4., limit_angle=45., backtrack=True,
                                            **GAPS))
        module = ns["module_specification"]
        array = resolve_array(ns["array_specification"], design)
        efficiency = ns["ENGINE_CONFIG"].conventional_peak_w/(1000*module.module_x*module.module_y)
        for hours in HOURS:
            print(f"  {mounting}, {hours} h/day (PV only, not cached)", flush=True)
            if dry:
                continue
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                hourly, _ = ns["model_for_optimization"](
                    design, ns["scenario"], module, array, calculate_linrui=False, calculate_hao=True,
                    sampling_config=SamplingConfig(mode="combined", days_per_month=1, hours_per_day=hours),
                    constructions_to_run=("conventional",))
            incident = sum(hourly[f"conventional_array_{f}_effective_incident_energy_Wh_hour"].sum()
                           for f in ("front", "rear"))
            kwp = array.n_mods*array.n_rows*ns["ENGINE_CONFIG"].conventional_peak_w/1000
            cfg = ns["ENGINE_CONFIG"]
            rows.append(dict(hours_per_day=hours, mounting=mounting,
                             kwh_per_kwp=incident*efficiency*cfg.temperature_factor*cfg.wiring_inverter_factor/1000/kwp))
            print(f"    {rows[-1]['kwh_per_kwp']:.0f} kWh/kWp in {(time.time()-t0)/60:.1f} min", flush=True)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.pivot(index="hours_per_day", columns="mounting", values="kwh_per_kwp")
    t["tracking_gain_%"] = 100*(t.single_axis/t.fixed_35deg-1)
    t["thesis_fixed"] = [HOURS_THESIS[h][0] for h in t.index]
    t["thesis_single_axis"] = [HOURS_THESIS[h][1] for h in t.index]
    return t.reset_index()


def main(argv):
    if not argv or argv[0] not in ("grid", "hours", "trace"):
        print(__doc__)
        return 2
    test, dry = argv[0], "--dry-run" in argv
    print(f"Accuracy test '{test}'{' (dry run)' if dry else ''}")
    table = {"grid": lambda: test_grid(dry), "hours": lambda: test_hours(dry),
             "trace": lambda: test_trace(dry, "--both" in argv)}[test]()
    if dry or table.empty:
        return 0
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT/f"{test}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    if test == "trace":
        print("Thesis:", TRACE_THESIS)
    print("Saved:", OUT/f"{test}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
