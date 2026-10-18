"""Water-limited check of the crop model (thesis Section 4.8), from the cached ground irradiation (no ray tracing).

The thesis runs WOFOST 7.2 in potential production (Wofost72_PP): water is never limiting. This script repeats
the crop simulation with the water-limited model Wofost72_WLP_CWB, driven by daily ERA5 rainfall for the source
months of the TMY (inputs/precipitation/, prepared by Linrui Li from Open-Meteo), for the open field and three
fixed-tilt Pareto designs (conventional modules, 192 skies), for six crops and two soils:
  deep    = the generic soil of bridge.run_crop (rooting depth 120 cm, 20 vol-% available water)
  shallow = assumption for a shallower, lighter soil (rooting depth 60 cm, 15 vol-% available water)
    python analysis/water_limited_check.py
Output: results/water_limited_check/water_limited_check.csv and summary.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

from common import RESULTS, ROOT

sys.path.insert(0, str(ROOT))
import bridge  # noqa: E402
import crops  # noqa: E402
from pcse.base import ParameterProvider  # noqa: E402
from pcse.input import WOFOST72SiteDataProvider  # noqa: E402
from pcse.models import Wofost72_PP, Wofost72_WLP_CWB  # noqa: E402
from sampling_core import SamplingConfig  # noqa: E402
from weighted_engine import make_engine  # noqa: E402

RAIN_FILE = ROOT/"inputs"/"precipitation"/"Aachen_TMY_precipitation_supplement.json"
SOILS = {"deep": dict(SMW=.10, SMFCF=.30, SM0=.45, CRAIRC=.06, SOPE=10., KSUB=10., RDMSOL=120.),  # = bridge.run_crop
         "shallow": dict(SMW=.10, SMFCF=.25, SM0=.45, CRAIRC=.06, SOPE=10., KSUB=10., RDMSOL=60.)}
CROPS = ["winter_wheat", "spring_barley", "grain_maize", "potato", "sugar_beet", "soybean"]
NOTEBOOK = "agripv_optimizer_fixed.ipynb"
SAMPLING = SamplingConfig(mode="combined", hours_per_day=8, days_per_month=2)


def notebook_namespace():
    nb = json.loads((ROOT/NOTEBOOK).read_text(encoding="utf-8"))
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
    return ns


def simulate(config, frame, irrad, model, soil):
    """Same set-up as bridge.run_crop (crop parameters, site, calendar, weather), with model and soil as inputs."""
    crop, _ = bridge.crop_parameters(config)
    site = dict(WOFOST72SiteDataProvider(WAV=20.))
    days = config.dates()
    calendar = [{days[0]: {"CropCalendar": dict(crop_name=config.crop_name, variety_name=config.variety,
                 crop_start_date=days[0], crop_start_type=config.start_type, crop_end_date=days[-1],
                 crop_end_type=config.end_type, max_duration=(days[-1]-days[0]).days),
                 "TimedEvents": None, "StateEvents": None}}]
    weather = bridge._weather_provider(frame, config, irrad)
    with contextlib.redirect_stdout(io.StringIO()):
        m = model(ParameterProvider(cropdata=crop, soildata=soil, sitedata=site), weather, calendar)
        m.run_till_terminate()
    return float(m.get_summary_output()[0]["TWSO"]), pd.DataFrame(m.get_output())


def main():
    ns = notebook_namespace()
    rain = {r["month_day"]: r["rain_mm_day"] for r in json.loads(RAIN_FILE.read_text(encoding="utf-8"))["records"]}
    pareto = pd.read_csv(RESULTS/"recheck_combined_192skies"/"pareto_front.csv", float_precision="round_trip")
    conv = pareto[pareto.pv_type == "conventional"].sort_values("LER")
    designs = {"2 rows": conv.loc[conv.annual_net_eur_ha.idxmax()], "3 rows": conv[conv.n_rows == 3].iloc[0],
               "5 rows": conv.iloc[-1]}
    rows = []
    for name in CROPS:
        crop_config, engine_config = crops.crop_setup(name, ns["ENGINE_CONFIG"])
        with contextlib.redirect_stdout(io.StringIO()):
            engine = make_engine(ns, crop_config, engine_config, ns["SAMPLING"], ns["REFERENCE_PV_DESIGN"],
                                 ns["PRICE_MODE"], ns["FLAT_PRICE_EUR_KWH"], ns["PRICE_YEARS"], parallel_designs=1,
                                 workers_per_design=None, notebook_path=NOTEBOOK, constructions=ns["CONSTRUCTIONS"])
        frame = engine.weather.copy()
        frame["RAIN_mm_day"] = [rain[f"{d.month:02d}-{d.day:02d}"] for d in frame.day]
        cases = {"open field": (None, frame.IRRAD.to_numpy())}
        for label, row in designs.items():
            design = engine._design_from_row(row)
            with contextlib.redirect_stdout(io.StringIO()):
                _, grounds, _ = engine._simulation(design, SAMPLING)
            ground = engine._crop_ground(grounds["conventional"])
            points = engine._grids(design)["conventional"].points()
            ground = ground.merge(points[["point_id", "cell_area_m2"]], on="point_id", suffixes=("_x", ""))
            weighted = (ground.total_Wh_m2_day*ground.cell_area_m2).groupby(ground.day).sum()
            cases[label] = (row, weighted.reindex(frame.day).to_numpy()/points.cell_area_m2.sum()*3600.)
        for label, (row, irrad) in cases.items():
            pp, daily_pp = simulate(crop_config, frame, irrad, Wofost72_PP, SOILS["deep"])
            out = dict(crop=name, case=label, n_rows=None if row is None else int(row.n_rows),
                       season_rain_mm=float(frame.RAIN_mm_day.sum()), TWSO_PP=pp)
            for soil, params in SOILS.items():
                wlp, daily = simulate(crop_config, frame, irrad, Wofost72_WLP_CWB, params)
                both = daily_pp[["day", "TRA"]].merge(daily[["day", "TRA"]], on="day", suffixes=("_pp", "_wlp"))
                out[f"TWSO_WLP_{soil}"] = wlp
                out[f"stress_days_{soil}"] = int((both.TRA_wlp < .999*both.TRA_pp).sum())
            rows.append(out)
        print(name, "done", flush=True)
    table = pd.DataFrame(rows)
    for col in ["TWSO_PP", "TWSO_WLP_deep", "TWSO_WLP_shallow"]:
        open_field = table[table.case == "open field"].set_index("crop")[col]
        table[col.replace("TWSO", "rel")] = table[col]/table.crop.map(open_field)
    out = RESULTS/"water_limited_check"
    out.mkdir(exist_ok=True)
    table.to_csv(out/"water_limited_check.csv", index=False)
    (out/"summary.json").write_text(json.dumps(dict(rain_file=str(RAIN_FILE.relative_to(ROOT)), soils=SOILS,
                                                   sampling="2 d x 8 h", designs={k: v.design_id for k, v in designs.items()}),
                                              indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
