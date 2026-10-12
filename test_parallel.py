"""Offline tests for parallel physics (mock physics, real subprocesses)."""
import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import bridge
from sampling_core import SamplingConfig, fingerprint
from weighted_engine import WeightedEngine, EngineConfig, _code_digest

ROOT = Path(__file__).resolve().parent

SETUP_CELL = '''
import sys
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import bridge
bridge.version = lambda package: "6.0.13"   # test env has no PCSE metadata
PROJECT_ROOT = Path(r"__ROOT__")
WORKING_DIRECTORY = PROJECT_ROOT

@dataclass(frozen=True)
class PVDesign:
    tilt: float = 25.
    pitch: float = 7.
    azimuth: float = 180.
    clearance_height: float = 3.
    xcellgap: float = .0004
    ycellgap: float = .00235
    xgap_conventional: float = .20

@dataclass(frozen=True)
class Scenario:
    start_date: str = "2001-01-01"
    end_date: str = "2001-12-31"
    weather_filename: str = "Aachen_PVGIS_TMY.epw"
    weather_time_basis: str = "local_standard"
    ground_grid_x: int = 2
    ground_grid_y: int = 2

@dataclass(frozen=True)
class Module:
    module_x: float = 1.134
    module_y: float = 2.278
    numcellsx: int = 6
    numcellsy: int = 24
    xcell: float = .182
    ycell: float = .091

@dataclass(frozen=True)
class Array:
    n_mods: int = 5
    n_rows: int = 5
    field_depth_m: float = 0.

@dataclass(frozen=True)
class Settings:
    ground_sensor_height: float = .05
    n_workers: int = 3

scenario, module_specification, array_specification, SETTINGS = Scenario(), Module(), Array(), Settings()
br = SimpleNamespace(RadianceObj=lambda **kw: None)

_hours = pd.date_range("2001-01-01", periods=365 * 24, freq="h", tz="Etc/GMT-1")
_ghi = np.maximum(0, np.sin((_hours.hour.to_numpy() - 6) * np.pi / 12)) * 500
_weather = pd.DataFrame({"GHI": _ghi, "DHI": _ghi * .3, "_crop_temp": 20., "_crop_dew": 10., "_crop_wind": 2.,
                         "_crop_rain": 0., "_crop_rain_period": 1}, index=_hours + pd.Timedelta(hours=1))
_crop_days = list(pd.date_range("2001-01-01", "2001-01-02").date)

def _dummy(*args, **kwargs):
    return None

for _name in ["make_hourly_sky", "create_selected_module", "analyze_array", "analyze_module", "analyze_ground_hour",
              "trace_direct_ground_irradiation", "calculate_hourly_results", "make_module_surface_scans",
              "make_rectangular_surface_scans", "module_geometry", "glass_margin_text", "add_hazy_glass_material"]:
    globals()[_name] = _dummy

def validate_cell_array_fit(design, module_spec):
    return None

def cultivation_period_bounds(sc):
    return pd.Timestamp(sc.start_date), pd.Timestamp(sc.end_date)

def read_hourly_cultivation_weather(demo, sc):
    global ALIGNED_HOURLY_WEATHER
    ALIGNED_HOURLY_WEATHER = _weather
    return _hours, _hours, None, {}

def model_for_optimization(design, sc, module, array, **kw):
    out = pd.DataFrame({"hour_start": [x.isoformat() for x in _hours]})
    grounds = {}
    for kind, grid in kw["ground_grid"].items():
        if kw["constructions_to_run"] and kind not in kw["constructions_to_run"]:
            continue
        out[f"{kind}_array_front_effective_incident_energy_Wh_hour"] = _ghi * (30 + design.pitch)
        out[f"{kind}_array_rear_effective_incident_energy_Wh_hour"] = _ghi * 3
        if kw["calculate_linrui"]:
            rows = []
            for day in _crop_days:
                irrad = _ghi[_hours.date == day].sum() * .8
                rows += [dict(day=day.isoformat(), total_Wh_m2_day=irrad, **p) for p in grid.points().to_dict("records")]
            grounds[kind] = pd.DataFrame(rows)
    return out, grounds
'''
DEAD_CELL = 'raise RuntimeError("worker must stop before the control panel")'


def build_namespace(root, notebook):
    ns = {"__name__": "__main__"}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "USER CONTROL PANEL" in source:
                break
            exec(compile(source, "<some-other-kernel-cell>", "exec"), ns)
    return ns


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/"weather").mkdir()
        shutil.copy(ROOT/"weather"/"Aachen_PVGIS_TMY.epw", self.root/"weather"/"Aachen_PVGIS_TMY.epw")
        cells = [dict(cell_type="code", metadata={}, outputs=[], execution_count=None,
                      source=SETUP_CELL.replace("__ROOT__", str(self.root)).splitlines(keepends=True)),
                 dict(cell_type="code", metadata={}, outputs=[], execution_count=None,
                      source=["# USER CONTROL PANEL\n", DEAD_CELL])]
        self.notebook = self.root/"fake.ipynb"
        self.notebook.write_text(json.dumps(dict(cells=cells, metadata={}, nbformat=4, nbformat_minor=5)))
        self.cfg = bridge.CropConfig(start="2001-01-01", end="2001-01-02", accept_demo_assumptions=True,
                                     allow_missing_rain_for_pp=True)
        self.ns = build_namespace(self.root, self.notebook)
        self.Design = self.ns["PVDesign"]

    def tearDown(self):
        self.tmp.cleanup()

    def engine(self, **kw):
        return WeightedEngine(self.ns, self.cfg, EngineConfig(), SamplingConfig("combined", 3, 2),
                              self.Design(pitch=3., clearance_height=.8), notebook_path="fake.ipynb", **kw)

    def test_code_digest_ignores_filename_and_line_numbers(self):
        source = "def f(x):\n    return x * 2 + 1\n"
        a, b = {}, {}
        exec(compile(source, "<kernel-cell-1>", "exec"), a)
        exec(compile("\n\n\n" + source, "/elsewhere/other.py", "exec"), b)
        self.assertEqual(_code_digest(a["f"].__code__), _code_digest(b["f"].__code__))
        c = {}
        exec(compile(source.replace("2", "3"), "<kernel-cell-1>", "exec"), c)
        self.assertNotEqual(_code_digest(a["f"].__code__), _code_digest(c["f"].__code__))

    def test_signature_ignores_thread_count(self):
        first = self.engine()
        sig = json.dumps(first.signature, sort_keys=True, default=str)
        self.ns["SETTINGS"] = self.ns["SETTINGS"].__class__(n_workers=17)
        self.assertEqual(sig, json.dumps(self.engine().signature, sort_keys=True, default=str))

    def test_precompute_plans_reference_first_and_dedupes(self):
        engine = self.engine(parallel_designs=3)
        a, b = self.Design(pitch=6.), self.Design(pitch=8.)
        with patch.object(WeightedEngine, "_run_jobs") as run:
            engine.precompute([a, b, a])
            jobs = run.call_args.args[0]
        self.assertEqual([k for k, _ in jobs], ["reference", "design", "design"])
        self.assertEqual([d.pitch for k, d in jobs if k == "design"], [6., 8.])

    def test_serial_mode_does_not_start_workers(self):
        engine = self.engine(parallel_designs=1)
        with patch.object(WeightedEngine, "_run_jobs") as run:
            engine.precompute([self.Design(pitch=6.)])
        run.assert_not_called()

    def test_real_worker_processes_share_cache_with_parent(self):
        engine = self.engine(parallel_designs=2, workers_per_design=2)
        designs = [self.Design(pitch=6.), self.Design(pitch=9.)]
        engine.precompute(designs)
        sampling = engine.sampling
        self.assertTrue(engine._physics_cached(engine.reference_design, sampling, crop=False))
        for design in designs:
            self.assertTrue(engine._physics_cached(design, sampling, crop=True))
        calls = []
        original = self.ns["model_for_optimization"]
        self.ns["model_for_optimization"] = lambda *a, **k: calls.append(1) or original(*a, **k)

        def mock_crop(frame, config, radiation):
            yield_dm = 10000*float(pd.Series(radiation).mean())/float(frame.IRRAD.mean())
            return dict(summary=dict(DOM=self.cfg.dates()[-1], TWSO=yield_dm, TAGP=2*yield_dm, DVS=2., CTRAT=1.),
                        mature=True, daily=pd.DataFrame({"day": self.cfg.dates()}),
                        weather=pd.DataFrame({"day": self.cfg.dates()}), parameters={}, soil={}, site={}, management=[])
        counter = [0]

        def new_run(project):
            counter[0] += 1
            path = self.root/f"crop{counter[0]}"
            path.mkdir()
            return path
        with patch.object(bridge, "run_crop", side_effect=mock_crop), patch.object(bridge, "new_run", side_effect=new_run):
            rows = engine.evaluate(designs[0])
        self.assertEqual(len(rows), 2)
        self.assertEqual(calls, [], "parent must reuse the workers' cache, not recompute physics")


    def test_nsga2_search_with_parallel_workers_end_to_end(self):
        engine = self.engine(parallel_designs=2, workers_per_design=2)

        def mock_crop(frame, config, radiation):
            yield_dm = 10000*float(pd.Series(radiation).mean())/float(frame.IRRAD.mean())
            return dict(summary=dict(DOM=self.cfg.dates()[-1], TWSO=yield_dm, TAGP=2*yield_dm, DVS=2., CTRAT=1.),
                        mature=True, daily=pd.DataFrame({"day": self.cfg.dates()}),
                        weather=pd.DataFrame({"day": self.cfg.dates()}), parameters={}, soil={}, site={}, management=[])
        counter = [0]

        def new_run(project):
            counter[0] += 1
            path = self.root/f"crop{counter[0]}"
            path.mkdir()
            return path
        calls = []
        original = self.ns["model_for_optimization"]
        self.ns["model_for_optimization"] = lambda *a, **k: calls.append(1) or original(*a, **k)
        with patch.object(bridge, "run_crop", side_effect=mock_crop), patch.object(bridge, "new_run", side_effect=new_run):
            table = engine.search(4, 2, 1, {"pitch": (4., 12.), "tilt": (10., 50.)}, self.Design())
        self.assertEqual(len(table.pv_type.unique()), 2)
        self.assertGreaterEqual(len(table), 8)
        self.assertEqual(calls, [], "all physics must have run in worker processes")
        self.assertTrue(list((engine.cache/"jobs").glob("design_*.log")))

    def test_signature_mismatch_is_reported_not_silently_cached(self):
        engine = self.engine(parallel_designs=1)
        # Parent edits a model function after the notebook was saved.
        self.ns["validate_cell_array_fit"] = self.ns["validate_cell_array_fit"]
        self.ns["make_hourly_sky"] = lambda *a, **k: 42   # different bytecode than the saved notebook
        engine = self.engine(parallel_designs=2)
        with self.assertRaises(RuntimeError) as caught:
            engine.precompute([self.Design(pitch=6.)])
        self.assertIn("exit code 3", str(caught.exception))
        self.assertIn("Save the notebook", str(caught.exception))


    # ---- fixed field (v6.4) -------------------------------------------------------------
    def fixed_field_engine(self, **kw):
        self.ns["array_specification"] = self.ns["Array"](n_mods=3, n_rows=1, field_depth_m=24.)
        return self.engine(**kw)

    def _mock_crop_patches(self):
        def mock_crop(frame, config, radiation):
            yield_dm = 10000*float(pd.Series(radiation).mean())/float(frame.IRRAD.mean())
            return dict(summary=dict(DOM=self.cfg.dates()[-1], TWSO=yield_dm, TAGP=2*yield_dm, DVS=2., CTRAT=1.),
                        mature=True, daily=pd.DataFrame({"day": self.cfg.dates()}),
                        weather=pd.DataFrame({"day": self.cfg.dates()}), parameters={}, soil={}, site={}, management=[])
        counter = [0]

        def new_run(project):
            counter[0] += 1
            path = self.root/f"crop{counter[0]}"
            path.mkdir()
            return path
        return (patch.object(bridge, "run_crop", side_effect=mock_crop),
                patch.object(bridge, "new_run", side_effect=new_run))

    def test_rows_follow_pitch_but_field_area_is_fixed(self):
        from sampling_core import resolve_array, design_grid
        array = self.ns["Array"](n_mods=3, n_rows=1, field_depth_m=24.)
        module = self.ns["module_specification"]
        rows = {pitch: resolve_array(array, self.Design(pitch=pitch)).n_rows for pitch in (4., 5., 6., 7., 8., 12.)}
        self.assertEqual(rows, {4.: 6, 5.: 4, 6.: 4, 7.: 3, 8.: 3, 12.: 2})
        areas = {round(design_grid(self.Design(pitch=p), module, array, "conventional").area_m2, 9) for p in rows}
        self.assertEqual(len(areas), 1, "the plot must not change with the pitch")
        self.assertEqual(resolve_array(array, self.Design(pitch=24.)).n_rows, 1)
        self.assertEqual(resolve_array(array, self.Design(pitch=40.)).n_rows, 1)   # never zero rows
        old = self.ns["Array"](n_mods=5, n_rows=5)   # field_depth_m == 0: old behaviour
        self.assertIs(resolve_array(old, self.Design(pitch=7.)), old)
        self.assertAlmostEqual(design_grid(self.Design(pitch=7.), module, old, "conventional").area_m2,
                               (5*module.module_x + 4*.2)*5*7.)

    def test_physics_and_economics_receive_the_derived_row_count(self):
        engine = self.fixed_field_engine()
        seen = []
        original = self.ns["model_for_optimization"]
        self.ns["model_for_optimization"] = lambda design, sc, module, array, **k: (seen.append(array.n_rows), original(design, sc, module, array, **k))[1]
        crop_a, crop_b = self._mock_crop_patches()
        with crop_a, crop_b:
            narrow = engine.evaluate(self.Design(pitch=4.))
            wide = engine.evaluate(self.Design(pitch=12.))
        # pitch 4 -> 6 rows; the reference (pitch 3) -> 8 rows, derived the same way; pitch 12 -> 2 rows
        self.assertEqual(seen, [6, 8, 2])
        self.assertEqual({r["n_rows"] for r in narrow}, {6})
        self.assertEqual({r["n_rows"] for r in wide}, {2})
        self.assertEqual({r["n_modules"] for r in narrow}, {18})
        self.assertEqual({r["n_modules"] for r in wide}, {6})
        self.assertAlmostEqual(narrow[0]["field_area_ha"], wide[0]["field_area_ha"])
        self.assertAlmostEqual(narrow[0]["field_area_ha"], 3*1.134*24/10000., places=6)

    def test_prices_and_costs_do_not_change_the_physics_key(self):
        base = self.engine()
        design = self.Design(pitch=6.)
        key = base._physics_key(design, base.sampling, crop=True)
        dearer = self.engine(price_mode="flat_assumption", flat_price=.12)
        self.assertEqual(dearer._physics_key(design, dearer.sampling, crop=True), key)
        self.assertNotEqual(fingerprint(dearer.signature), fingerprint(base.signature))
        eval_a = base._evaluation_cached(design, base.sampling)
        self.assertFalse(eval_a)
        costly = WeightedEngine(self.ns, self.cfg, EngineConfig(module_semi_eur_kwp=999.), SamplingConfig("combined", 3, 2),
                                self.Design(pitch=3., clearance_height=.8), notebook_path="fake.ipynb")
        self.assertEqual(costly._physics_key(design, costly.sampling, crop=True), key)
        # ...but anything that changes the ray tracing must change it
        traced = WeightedEngine(self.ns, self.cfg, EngineConfig(trace_ground_components=True), SamplingConfig("combined", 3, 2),
                                self.Design(pitch=3., clearance_height=.8), notebook_path="fake.ipynb")
        self.assertNotEqual(traced._physics_key(design, traced.sampling, crop=True), key)
        self.assertNotEqual(self.fixed_field_engine()._physics_key(design, base.sampling, crop=True), key)
        self.assertNotEqual(base._physics_key(self.Design(pitch=7.), base.sampling, crop=True), key)

    def test_pitch_bound_larger_than_field_depth_is_rejected(self):
        engine = self.fixed_field_engine()
        with self.assertRaises(ValueError) as caught:
            engine.search(4, 1, 1, {"pitch": (4., 30.)}, self.Design())
        self.assertIn("field depth", str(caught.exception))

    def test_worker_accepts_a_price_change_because_it_checks_the_physics_signature(self):
        engine = self.engine(parallel_designs=2, workers_per_design=2, price_mode="flat_assumption", flat_price=.11)
        engine.precompute([self.Design(pitch=6.)])
        self.assertTrue(engine._physics_cached(self.Design(pitch=6.), engine.sampling, crop=True))

    def test_stage2_recheck_reevaluates_search_designs_with_denser_sampling(self):
        engine = self.engine(parallel_designs=2, workers_per_design=2)
        crop_a, crop_b = self._mock_crop_patches()
        calls = []
        original = self.ns["model_for_optimization"]
        self.ns["model_for_optimization"] = lambda *a, **k: calls.append(1) or original(*a, **k)
        with crop_a, crop_b:
            found = engine.search(4, 2, 1, {"pitch": (4., 12.), "tilt": (10., 50.)}, self.Design())
            pareto = found[found.pareto & found.feasible]
            dense = SamplingConfig("combined", 4, 3)
            self.assertGreater(engine.n_skies(dense), engine.n_skies(engine.sampling))
            table = engine.recheck(pareto, dense)
        self.assertEqual(calls, [], "re-check physics must also run in worker processes")
        # every unique search design is evaluated for BOTH constructions
        self.assertEqual(set(table.design_id), set(pareto.design_id))
        for _, group in table.groupby("design_id"):
            self.assertEqual(len(group), 2)
        self.assertEqual(set(table.n_skies), {engine.n_skies(dense)})
        self.assertIn("pareto", table.columns)
        folder = engine.root/"results"/f"recheck_combined_{engine.n_skies(dense)}skies"
        for name in ("evaluated_candidates.csv", "pareto_front.csv", "comparison_with_search.csv", "run_metadata.json"):
            self.assertTrue((folder/name).exists(), name)
        comparison = pd.read_csv(folder/"comparison_with_search.csv")
        self.assertIn("LER_change_%", comparison.columns)
        self.assertEqual(len(comparison), len(pareto.drop_duplicates(["design_id", "pv_type"])))
        with self.assertRaises(ValueError):
            engine.recheck(pareto.drop(columns=["tilt"]), dense)


class ConstraintAndToolingTests(unittest.TestCase):
    def test_constraint_violation_is_continuous_and_zero_when_feasible(self):
        import pandas as pd
        from types import SimpleNamespace
        from weighted_engine import economic_metrics
        from sampling_core import design_grid
        module = SimpleNamespace(module_x=1.134, module_y=2.278, numcellsx=6, numcellsy=24, xcell=.182, ycell=.091)
        array = SimpleNamespace(n_mods=5, n_rows=5)
        from dataclasses import make_dataclass
        D = make_dataclass("D", [("tilt", float), ("pitch", float), ("azimuth", float), ("clearance_height", float),
                                 ("xcellgap", float), ("ycellgap", float), ("xgap_conventional", float)])
        hours = pd.date_range("2001-10-01", periods=8760, freq="h")
        hourly = pd.DataFrame({"semi_transparent_array_front_effective_incident_energy_Wh_hour": 100.,
                               "semi_transparent_array_rear_effective_incident_energy_Wh_hour": 10.}, index=hours)
        def metrics(crop_yield, pitch=8., clearance=3.):
            design = D(25., pitch, 180., clearance, .0004, .00235, .2)
            grid = design_grid(design, module, array, "semi_transparent")
            return economic_metrics("semi_transparent", hourly, [.08]*8760, design, module, array, grid,
                                    crop_yield, 8., True, 500., EngineConfig())
        ok = metrics(8.)
        self.assertTrue(ok["feasible"])
        self.assertEqual(ok["constraint_violation"], 0.)
        worse, worst = metrics(4.), metrics(1.)
        self.assertFalse(worse["feasible"])
        self.assertGreater(worse["constraint_violation"], 0.)
        self.assertGreater(worst["constraint_violation"], worse["constraint_violation"])
        self.assertGreater(metrics(8., clearance=2.)["constraint_violation"], 0.)

    def test_analyze_ground_hour_skips_direct_trace_on_request(self):
        import ast
        import numpy as np
        notebook = json.loads((ROOT/"agripv_optimizer_fixed.ipynb").read_text())
        tree = ast.parse("".join(notebook["cells"][8]["source"]))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "analyze_ground_hour")
        traced = []

        class Analysis:
            def __init__(self, *a, **k): pass
            def analysis(self, octfile, name, scan, accuracy=None): return {"Wm2": [1., 2., 3.]}
        from types import SimpleNamespace
        ns = dict(np=np, br=SimpleNamespace(AnalysisObj=Analysis), SETTINGS=SimpleNamespace(hpc=True, accuracy="low"),
                  trace_direct_ground_irradiation=lambda *a: traced.append(1) or np.array([.5, .5, .5]))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "nb", "exec"), ns)
        direct, diffuse, total = ns["analyze_ground_hour"]("o", "s", {}, "a", with_components=False)
        self.assertEqual(traced, [])
        np.testing.assert_array_equal(total, [1., 2., 3.])
        self.assertTrue(np.isnan(direct).all())
        direct, diffuse, total = ns["analyze_ground_hour"]("o", "s", {}, "a")
        self.assertEqual(traced, [1])
        np.testing.assert_allclose(diffuse, [.5, 1.5, 2.5])

    def _array_function(self, mode, n_mods=5, n_rows=5):
        import ast
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        notebook = json.loads((ROOT/"agripv_optimizer_fixed.ipynb").read_text())
        tree = ast.parse("".join(notebook["cells"][8]["source"]))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "analyze_array")
        traced = []

        def analyze_module(position, *args):
            traced.append(position)
            return 10., 2.            # every module delivers the same power
        ns = dict(ThreadPoolExecutor=ThreadPoolExecutor, analyze_module=analyze_module,
                  SETTINGS=SimpleNamespace(n_workers=2, pv_trace_mode=mode),
                  SimulationContext=object, PVDesign=object, ModuleSpecification=object, ArraySpecification=object)
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "nb", "exec"), ns)
        return ns["analyze_array"], traced, SimpleNamespace(n_mods=n_mods, n_rows=n_rows)

    def test_center_mode_traces_fewer_modules_and_scales_to_full_array(self):
        function, traced, array = self._array_function("all")
        self.assertEqual(function(None, None, None, array), (250., 50.))
        self.assertEqual(len(traced), 25)
        function, traced, array = self._array_function("center")
        front, rear = function(None, None, None, array)
        self.assertEqual(sorted(traced), [(3, 2), (3, 3), (3, 4)])
        self.assertAlmostEqual(front, 250.)   # same total when all modules are alike
        self.assertAlmostEqual(rear, 50.)

    def test_three_module_row_traces_exactly_one_module(self):
        function, traced, array = self._array_function("center", n_mods=3, n_rows=4)
        front, rear = function(None, None, None, array)
        self.assertEqual(traced, [(2, 2)])
        self.assertAlmostEqual(front, 120.)    # 12 modules x 10
        self.assertAlmostEqual(rear, 24.)
        function, traced, array = self._array_function("center", n_mods=3, n_rows=1)
        self.assertEqual(function(None, None, None, array), (30., 6.))
        self.assertEqual(traced, [(1, 2)])

    def test_center_mode_handles_small_arrays_and_rejects_unknown_mode(self):
        function, traced, array = self._array_function("center", n_mods=2, n_rows=1)
        front, _ = function(None, None, None, array)
        self.assertEqual(sorted(traced), [(1, 1), (1, 2)])   # no interior module: use the whole row
        self.assertAlmostEqual(front, 20.)
        function, _, array = self._array_function("bogus")
        with self.assertRaises(ValueError):
            function(None, None, None, array)

    def test_kill_tree_stops_a_running_process(self):
        import os
        import subprocess
        import time
        from weighted_engine import _kill_tree
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                start_new_session=(os.name != "nt"))
        t0 = time.time()
        _kill_tree(proc)
        proc.wait(timeout=10)
        self.assertLess(time.time()-t0, 10)
        self.assertIsNotNone(proc.poll())


if __name__ == "__main__":
    unittest.main()
