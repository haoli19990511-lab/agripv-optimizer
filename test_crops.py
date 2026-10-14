"""Crop selection: presets, crop-independent physics cache, calendar wrap-around (no Radiance run).
    python -m unittest test_crops -v
"""
import json
import unittest
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import bridge
import crops
from sampling_core import SamplingConfig, fingerprint
from weighted_engine import EngineConfig, make_engine

ROOT = Path(__file__).resolve().parent
WEATHER = ROOT/"weather"/"Aachen_PVGIS_TMY.epw"


def notebook_namespace(name="agripv_optimizer_fixed.ipynb"):
    """Model cells and control panel of the notebook (search switched off), as analysis/run_search.py does."""
    ns = {"__name__": "__main__", "display": lambda *a, **k: None}
    for cell in json.loads((ROOT/name).read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "USER CONTROL PANEL" in source:
            exec(compile(source.replace("RUN_SEARCH = True", "RUN_SEARCH = False"), "<control-panel>", "exec"), ns)
            break
        exec(compile(source, "<notebook-cell>", "exec"), ns)
    assert ns["RUN_SEARCH"] is False
    return ns


class PresetTests(unittest.TestCase):
    def test_every_preset_completes_its_season_in_the_open_field(self):
        for key in crops.CROP_PRESETS:
            config, engine_config = crops.crop_setup(key, EngineConfig())
            frame, _ = bridge.read_epw_daily(WEATHER, config)
            result = bridge.run_crop(frame, config)
            with self.subTest(crop=key):
                self.assertTrue(result["mature"])
                self.assertGreater(result["summary"]["TWSO"], 1000.)
                self.assertEqual(engine_config.crop_price_eur_t, crops.CROP_PRESETS[key].price_eur_t)

    def test_winter_wheat_preset_equals_the_previous_configuration(self):
        old = bridge.CropConfig(variety="Winter_wheat_103", start="2001-10-15", end="2002-08-31",
                                start_type="sowing", accept_demo_assumptions=True, allow_missing_rain_for_pp=True)
        new, engine_config = crops.crop_setup("winter_wheat", EngineConfig())
        self.assertEqual(old, new)
        self.assertEqual(asdict(engine_config), asdict(EngineConfig()))
        self.assertEqual(bridge.crop_file_path(new).read_bytes().replace(b"\r\n", b"\n"),
                         (ROOT/"inputs"/"wheat.yaml").read_bytes().replace(b"\r\n", b"\n"))

    def test_overrides_and_unknown_variety(self):
        config, _ = crops.crop_setup("sugar_beet", EngineConfig(), variety="Achat", end_type="maturity")
        self.assertEqual((config.variety, config.end_type), ("Achat", "maturity"))
        with self.assertRaises(ValueError):
            crops.crop_setup("potato", EngineConfig(), variety="not_a_variety")
        with self.assertRaises(ValueError):
            crops.crop_setup("banana", EngineConfig())


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = notebook_namespace()
        ns = cls.ns
        cls.engines = {}
        for key in ("winter_wheat", "grain_maize"):
            config, engine_config = crops.crop_setup(key, ns["ENGINE_CONFIG"])
            cls.engines[key] = make_engine(ns, config, engine_config, SamplingConfig(mode="combined", days_per_month=1,
                                           hours_per_day=2), ns["REFERENCE_PV_DESIGN"], "csv", None, [2023, 2024, 2025])

    def test_physics_cache_key_does_not_depend_on_the_crop(self):
        wheat, maize = self.engines["winter_wheat"], self.engines["grain_maize"]
        self.assertEqual(fingerprint(wheat.physics_signature), fingerprint(maize.physics_signature))
        design = self.ns["BASE_DESIGN"]
        sampling = wheat.sampling
        self.assertEqual(wheat._physics_key(design, sampling), maize._physics_key(design, sampling))
        self.assertNotEqual(fingerprint(wheat.signature), fingerprint(maize.signature))

    def test_ground_is_traced_for_the_whole_pv_year(self):
        engine = self.engines["winter_wheat"]
        self.assertEqual(len(engine.pv_days), 365)

    def test_crop_days_after_the_pv_year_wrap_to_the_same_month_and_day(self):
        maize = self.engines["grain_maize"]
        last = maize.crop_days[-1]
        self.assertGreater(last, maize.pv_days[-1])               # 31 Oct 2002, after the PV year
        self.assertEqual(maize._pv_day[last], last.replace(year=last.year - 1))
        # Synthetic full-year ground: value = day of year, so the mapping can be checked exactly.
        ground = pd.DataFrame([dict(day=d, pv_type="conventional", point_id=p, x_m=0., y_m=0., cell_area_m2=1.,
                                    total_Wh_m2_day=float(d.timetuple().tm_yday)) for d in maize.pv_days for p in (0, 1)])
        crop = maize._crop_ground(ground)
        self.assertEqual(sorted(crop.day.unique()), maize.crop_days)
        row = crop[(crop.day == last) & (crop.point_id == 0)].iloc[0]
        self.assertEqual(row.total_Wh_m2_day, float(last.timetuple().tm_yday))
        # The crop weather uses the same month and day of the typical year.
        self.assertEqual(list(maize.weather.day), maize.crop_days)

    def test_result_folders_keep_wheat_names_and_separate_other_crops(self):
        self.assertTrue(str(self.engines["winter_wheat"]._results_folder("nsga2_combined")).endswith("nsga2_combined"))
        self.assertTrue(str(self.engines["grain_maize"]._results_folder("nsga2_combined"))
                        .endswith("nsga2_combined_maize_Grain_maize_201"))


if __name__ == "__main__":
    unittest.main()
