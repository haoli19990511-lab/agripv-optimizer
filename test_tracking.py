"""Offline checks of the v7 single-axis additions (no Radiance, no WOFOST).
    python -m unittest -v test_tracking.py
"""
import ast
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pandas as pd

from weighted_engine import EngineConfig, WeightedEngine, build_design, economic_metrics, mounting_geometry

ROOT = Path(__file__).resolve().parent
MODULE = SimpleNamespace(module_x=1.134, module_y=2.278, module_thickness=.03)


def notebook_cells():
    nb = json.loads((ROOT/"agripv_optimizer_single_axis.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def design_classes():
    """PVDesign and TrackingDesign exactly as defined in the notebook."""
    code = next(c for c in notebook_cells() if "class TrackingDesign" in c)
    tree = ast.parse(code)
    keep = [n for n in tree.body if (isinstance(n, ast.ClassDef) and n.name in
                                     ("PVDesign", "TrackingDesign", "VerticalPVDesign"))
            or (isinstance(n, ast.ImportFrom) and n.module == "dataclasses")]
    ns = {"dataclass": dataclass}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<classes>", "exec"), ns)
    return ns["PVDesign"], ns["TrackingDesign"], ns["VerticalPVDesign"]


PVDesign, TrackingDesign, VerticalPVDesign = design_classes()
GAPS = dict(xcellgap=.0004, ycellgap=.00235, xgap_conventional=.2)


class TrackingTests(unittest.TestCase):
    def test_fixed_geometry_is_the_v6_formula(self):
        d = PVDesign(tilt=30., pitch=5., azimuth=180., clearance_height=3.2, **GAPS)
        g = mounting_geometry(d, MODULE)
        self.assertEqual(g["mounting"], "fixed")
        self.assertEqual(g["structure_height_m"], 3.2)
        self.assertEqual(g["min_clearance_m"], 3.2)
        self.assertAlmostEqual(g["clear_corridor_m"], 5.-2.278*np.cos(np.radians(30.)))

    def test_tracking_geometry_at_rotation_limit(self):
        d = TrackingDesign(pitch=6., axis_azimuth=180., hub_height=4., limit_angle=45., backtrack=True, **GAPS)
        g = mounting_geometry(d, MODULE)
        a = np.radians(45.)
        self.assertEqual(g["mounting"], "single_axis")
        self.assertEqual(g["structure_height_m"], 4.)
        self.assertAlmostEqual(g["min_clearance_m"], 4.-.5*(2.278*np.sin(a)+.03*np.cos(a)))
        self.assertAlmostEqual(g["clear_corridor_m"], 6.-2.278*np.cos(a))

    def test_grid_orientation_follows_rows(self):
        # Axis north-south (180) -> rows run north-south -> plot oriented like a fixed rack facing east (90).
        for axis, facing in ((180., 90.), (90., 0.), (0., 270.)):
            d = TrackingDesign(pitch=6., axis_azimuth=axis, hub_height=4., limit_angle=45., backtrack=True, **GAPS)
            self.assertAlmostEqual(d.azimuth, facing)

    def test_tracking_cell_parses_and_defines_functions(self):
        code = next(c for c in notebook_cells() if "def calculate_hourly_results_tracking" in c)
        names = {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
        for f in ("calculate_hourly_results_tracking", "analyze_tracking_array", "analyze_tracking_module",
                  "make_tracking_surface_scans", "make_tracking_rectangular_scans",
                  "validate_tracking_design", "tracking_min_clearance"):
            self.assertIn(f, names)
        # Fixed-tilt function names are not redefined (their cache digests must stay unchanged).
        for f in ("calculate_hourly_results", "analyze_array", "analyze_module", "make_module_surface_scans"):
            self.assertNotIn(f, names)

    def test_design_from_row_uses_the_mounting_class(self):
        engine = WeightedEngine.__new__(WeightedEngine)
        engine.ns = {"PVDesign": PVDesign, "TrackingDesign": TrackingDesign}
        engine.mounting = "single_axis"
        row = dict(pitch="6.5", axis_azimuth=170, hub_height=4.2, limit_angle=50, backtrack=True, **GAPS)
        d = engine._design_from_row(row)
        self.assertIsInstance(d, TrackingDesign)
        self.assertIs(d.backtrack, True)
        self.assertEqual(d.pitch, 6.5)
        engine.mounting = "fixed"
        d = engine._design_from_row(dict(tilt=30, pitch=5, azimuth=180, clearance_height=3, **GAPS))
        self.assertIsInstance(d, PVDesign)

    def test_tracker_premium_in_default_config(self):
        c = EngineConfig()
        self.assertEqual(c.tracker_extra_eur_kwp, 150.)
        self.assertEqual(c.other_capex_eur_kwp, 650.)  # literature CAPEX (Dimitris, 2026-10-11/12)
        self.assertEqual(c.structure_base_eur_kwp, 300.)


class VerticalTests(unittest.TestCase):
    def _economics(self, design, n_rows=3, depth=24.):
        module = SimpleNamespace(module_x=1.134, module_y=2.278, module_thickness=.03, numcellsx=6, numcellsy=24,
                                 xcell=.182, ycell=.091)
        hourly = pd.DataFrame({f"conventional_array_{f}_effective_incident_energy_Wh_hour": np.full(8760, 100.)
                               for f in ("front", "rear")})
        array = SimpleNamespace(n_mods=3, n_rows=n_rows, field_depth_m=depth)
        grid = SimpleNamespace(area_m2=3*1.334*depth)
        return economic_metrics("conventional", hourly, np.full(8760, .06), design, module, array, grid,
                                8., 9., True, 1.0e6, EngineConfig())

    def test_vertical_class_fixes_tilt(self):
        d = VerticalPVDesign(pitch=8., azimuth=90., clearance_height=.6, **GAPS)
        self.assertEqual(d.tilt, 90.)
        rebuilt = build_design(VerticalPVDesign, dict(vars(d)))  # tilt in the row is ignored
        self.assertEqual(rebuilt, d)

    def test_vertical_skips_overhead_clearance_and_uses_low_structure(self):
        d = VerticalPVDesign(pitch=8., azimuth=90., clearance_height=.6, **GAPS)
        g = mounting_geometry(d, MODULE)
        self.assertEqual(g["mounting"], "vertical")
        self.assertFalse(g["clearance_applies"])
        self.assertAlmostEqual(g["clear_corridor_m"], 8.-.03)
        row = self._economics(d)
        self.assertTrue(row["feasible"], row["infeasible_reason"])
        self.assertAlmostEqual(row["capex_total_eur"], row["installed_kwp"]*(220.+50.+650.))

    def test_vertical_planted_area_loses_one_strip_per_row(self):
        d = VerticalPVDesign(pitch=8., azimuth=90., clearance_height=.6, **GAPS)
        row = self._economics(d, n_rows=3, depth=24.)
        self.assertAlmostEqual(row["planted_area_ha"]/row["footprint_ha"], 1.-3*.5/24.)
        self.assertAlmostEqual(row["LER_crop"], 8./9.*(1.-3*.5/24.))

    def test_fixed_economics_unchanged(self):
        d = PVDesign(tilt=30., pitch=8., azimuth=180., clearance_height=2., **GAPS)
        row = self._economics(d)
        self.assertEqual(row["infeasible_reason"], "insufficient_clearance")
        self.assertAlmostEqual(row["planted_area_ha"]/row["footprint_ha"], .97)
        self.assertAlmostEqual(row["capex_total_eur"], row["installed_kwp"]*(220.+300.+650.))


if __name__ == "__main__":
    unittest.main(verbosity=2)
