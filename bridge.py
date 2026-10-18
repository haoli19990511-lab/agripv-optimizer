"""Crop-model interface (Linrui Li's WOFOST 7.2 / PCSE setup): weather conversion, crop library and simulation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from copy import deepcopy
from importlib.metadata import version
import csv
import hashlib
import json
import math
import os
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
ADAPTER_VERSION = "crop-bridge-1.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save_json(path, value):
    def serializable(item):
        if isinstance(item, dict):
            return {str(k): serializable(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [serializable(v) for v in item]
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(serializable(value), ensure_ascii=False, indent=2, default=str, allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def new_run(project):
    # A new folder for every run: never overwrite the official website's records.
    path = ROOT / "runs" / project / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_pcse():
    if version("pcse") != "6.0.13":
        raise RuntimeError("请使用 pcse==6.0.13，与网站已经验证的模型版本保持一致。")
    # Matches the website's isolated first-import behavior. Never change HOME.
    if "pcse" not in sys.modules:
        from unittest.mock import patch
        runtime = ROOT / "runs" / "pcse_runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        expanduser = os.path.expanduser
        with patch("os.path.expanduser", lambda p: str(runtime) if p == "~" else expanduser(p)):
            import pcse
    import pcse
    return pcse


def official_check():
    """Numerically reproduce the same six values and two checkpoints as the website."""
    pcse = load_pcse()
    model = pcse.start_wofost(grid=31031, crop=1, year=2000, mode="pp")
    model.run()
    model.run(days=10)
    lai11 = model.get_variable("LAI")
    model.run(days=25)
    lai36 = model.get_variable("LAI")
    model.run_till_terminate()
    daily, summary = pd.DataFrame(model.get_output()), model.get_summary_output()[0]
    expected = dict(TWSO=8729.399812508796, TAGP=18091.006102200274,
                    LAIMAX=6.2324058612208715, DVS=2.0,
                    CTRAT=26.975472103613182, CEVST=6.173597870892306)
    checks = {k: math.isclose(summary[k], v, rel_tol=1e-9, abs_tol=1e-8) for k, v in expected.items()}
    checks.update(days=len(daily) == 152, maturity=str(summary["DOM"]) == "2000-05-31",
                  LAI11=math.isclose(lai11, .2870809817505803, rel_tol=1e-9),
                  LAI36=math.isclose(lai36, 1.5281221557442746, rel_tol=1e-9))
    if not all(checks.values()):
        raise AssertionError(f"官方数值复现未通过：{checks}")
    folder = new_run("official_wheat_readonly_benchmark")
    daily.to_csv(folder / "daily.csv", index=False, encoding="utf-8-sig")
    save_json(folder / "validation.json", dict(checks=checks, summary=summary, pcse=pcse.__version__))
    return daily, summary, folder


@dataclass(frozen=True)
class CropConfig:
    # This is an explicit illustrative Aachen configuration, NOT a calibrated local trial.
    variety: str = "Winter_wheat_103"
    start: str = "2001-10-15"
    end: str = "2002-08-31"
    start_type: str = "sowing"
    accept_demo_assumptions: bool = False
    allow_missing_rain_for_pp: bool = False
    epw_wind_height_m: float = 10.0
    angstrom_a: float = .29
    angstrom_b: float = .49
    # Crop selection (WOFOST 7.2 parameter library in inputs/crops). crop_name is the PCSE crop label,
    # crop_file the YAML file with the variety; end_type "maturity" ends at DVS 2 (at the latest on `end`),
    # "harvest" ends on `end` (crops such as sugar beet that are harvested before maturity).
    crop_name: str = "wheat"
    crop_file: str = "wheat.yaml"
    end_type: str = "maturity"

    def dates(self):
        first, last = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if last <= first or (last - first).days > 365:
            raise ValueError("作物季至少应有两天，日期顺序正确，长度不超过 366 天。")
        days = list(pd.date_range(first, last).date)
        if any(d.month == 2 and d.day == 29 for d in days):
            raise ValueError("当前 TMY 是 365 天；此适配器不擅自补造 2 月 29 日天气。")
        if self.start_type not in ("sowing", "emergence"):
            raise ValueError("start_type 只能是 sowing（播种）或 emergence（出苗）。")
        if self.end_type not in ("maturity", "harvest"):
            raise ValueError("end_type must be 'maturity' or 'harvest'")
        if self.epw_wind_height_m not in (2.0, 10.0):
            raise ValueError("当前仅实现 2 m 原值或 10 m → 2 m 的风速转换。")
        return days


def read_epw_daily(path, config):
    """EPW hour 1..24 belongs to its printed month/day; never average a monthly total."""
    days = config.dates()
    path = Path(path)
    with path.open(encoding="utf-8-sig") as handle:
        loc = next(csv.reader(handle))
    if loc[0] != "LOCATION":
        raise ValueError("EPW 缺少 LOCATION 表头。")
    latitude, longitude, tz, elevation = map(float, loc[6:10])
    h = pd.read_csv(path, skiprows=8, header=None)
    if len(h) != 8760 or h.shape[1] < 35:
        raise ValueError("目前仅接受完整 8760 小时的标准 EPW TMY，不自动填补缺失记录。")
    fields = {"month": 1, "day": 2, "hour": 3, "temp": 6, "dew": 7,
              "ghi": 13, "dni": 14, "dhi": 15, "wind": 21, "rain": 33, "rain_period": 34}
    hourly = pd.DataFrame({name: pd.to_numeric(h[col], errors="raise") for name, col in fields.items()})
    if not np.isfinite(hourly.to_numpy()).all():
        raise ValueError("EPW 包含非有限数值。")
    if ((hourly.temp >= 99.9) | (hourly.dew >= 99.9) | (hourly.wind >= 999) |
        (hourly[["ghi", "dni", "dhi"]] >= 9999).any(axis=1)).any():
        raise ValueError("EPW 温度、风速或辐射包含标准缺失标记，不能运行。")
    if (hourly[["ghi", "dni", "dhi", "wind"]] < 0).any().any():
        raise ValueError("EPW 辐射或风速为负数。")
    # Daily mean actual vapour pressure from hourly dewpoint (hPa).
    hourly["vap"] = 6.108 * np.exp(17.27 * hourly.dew / (hourly.dew + 237.3))
    grouped = {}
    for key, group in hourly.groupby(["month", "day"]):
        if sorted(group.hour.tolist()) != list(range(1, 25)):
            raise ValueError(f"{key} 的小时重复或缺失。")
        valid_rain = bool(((group.rain < 999) & (group.rain >= 0) & (group.rain_period == 1)).all())
        grouped[key] = dict(TMIN=float(group.temp.min()), TMAX=float(group.temp.max()),
                            TEMP=float(group.temp.mean()), VAP=float(group.vap.mean()),
                            WIND=float(group.wind.mean()), IRRAD=float(group.ghi.sum()) * 3600,
                            DHI_Wh_m2_day=float(group.dhi.sum()),
                            RAIN_mm_day=float(group.rain.sum()) if valid_rain else None)
    records = []
    for day in days:
        if (day.month, day.day) not in grouped:
            raise ValueError(f"TMY 缺少 {day:%m-%d}。")
        records.append(dict(day=day, LAT=latitude, LON=longitude, ELEV=elevation,
                            **grouped[(day.month, day.day)]))
    frame = pd.DataFrame(records)
    metadata = dict(epw_sha256=sha256(path), latitude=latitude, longitude=longitude,
                    elevation_m=elevation, time_zone_utc_offset=tz, weather_type="TMY, not observed 2001/2002",
                    rainfall_missing_days=int(frame.RAIN_mm_day.isna().sum()),
                    epw_wind_height_assumed_m=config.epw_wind_height_m,
                    vapour_pressure_method="mean of hourly dewpoint-derived vapour pressure")
    return frame, metadata


CROP_LIBRARY = ROOT / "inputs" / "crops"  # official WOFOST 7.2 parameter sets, see PROVENANCE.json


def crop_file_path(config):
    path = CROP_LIBRARY / config.crop_file
    if not path.exists():
        raise ValueError(f"Unknown crop file {config.crop_file}; available: "
                         f"{sorted(p.name for p in CROP_LIBRARY.glob('*.yaml') if p.name != 'crops.yaml')}")
    return path


def crop_varieties(crop_file):
    """Parameter sets (varieties) of one crop file of the library, with their description."""
    content = yaml.safe_load((CROP_LIBRARY / crop_file).read_text(encoding="utf-8"))
    return {name: raw.get("Metadata", {}).get("Description", "").strip()
            for name, raw in content["CropParameters"]["Varieties"].items()}


def crop_parameters(config):
    path = crop_file_path(config)
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    varieties = content["CropParameters"]["Varieties"]
    if config.variety not in varieties:
        raise ValueError(f"未知品种。可选：{list(varieties)}")
    raw = varieties[config.variety]
    parameters = {k: deepcopy(v[0]) for k, v in raw.items() if k != "Metadata"}
    return parameters, dict(variety=config.variety, metadata=raw["Metadata"],
                            yaml_sha256=sha256(path), parameters=parameters)


def _weather_provider(frame, config, radiation):
    load_pcse()
    from pcse.base import WeatherDataProvider, WeatherDataContainer
    from pcse.util import reference_ET, wind10to2
    missing = frame.RAIN_mm_day.isna().any()
    if missing and not config.allow_missing_rain_for_pp:
        raise ValueError("EPW 降雨缺失。若仅测试无水分限制的 PP 产量，请明确允许占位；不能据此预测灌溉或土壤蒸发。")
    values = np.asarray(radiation, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("IRRAD 必须为完整逐日、非负、有限的 J/m²/day。")
    provider = WeatherDataProvider()
    provider.latitude, provider.longitude, provider.elevation = map(float, frame.iloc[0][["LAT", "LON", "ELEV"]])
    provider.description = ["Aachen EPW + reviewed daily radiation adapter; WOFOST72 PP only"]
    provider.angstA, provider.angstB = config.angstrom_a, config.angstrom_b
    for row, irrad in zip(frame.to_dict("records"), values):
        wind = wind10to2(row["WIND"]) if config.epw_wind_height_m == 10 else row["WIND"]
        e0, es0, et0 = reference_ET(row["day"], row["LAT"], row["ELEV"], row["TMIN"], row["TMAX"],
                                   float(irrad), row["VAP"], wind, config.angstrom_a, config.angstrom_b)
        provider._store_WeatherDataContainer(WeatherDataContainer(
            DAY=row["day"], LAT=row["LAT"], LON=row["LON"], ELEV=row["ELEV"],
            IRRAD=float(irrad), TMIN=row["TMIN"], TMAX=row["TMAX"], TEMP=row["TEMP"],
            VAP=row["VAP"], WIND=wind, E0=e0 / 10, ES0=es0 / 10, ET0=et0 / 10,
            RAIN=0.0 if pd.isna(row["RAIN_mm_day"]) else row["RAIN_mm_day"] / 10), row["day"])
    return provider


def run_crop(frame, config, radiation=None):
    """Real PCSE simulation; only IRRAD and derived E0/ES0/ET0 change by scenario."""
    if not config.accept_demo_assumptions:
        raise ValueError("请先阅读并确认 Aachen 示例品种、播种日期和非当地标定土壤假设。")
    expected_days = config.dates()
    if list(frame.day) != expected_days:
        raise ValueError("天气与配置日期必须逐日完全匹配，不能跳日或填充。")
    load_pcse()
    from pcse.base import ParameterProvider
    from pcse.models import Wofost72_PP
    from pcse.input import WOFOST72SiteDataProvider
    crop, parameter_info = crop_parameters(config)
    # Generic PP demonstration soil, fully recorded; NOT Aachen measurements.
    soil = dict(SMW=.10, SMFCF=.30, SM0=.45, CRAIRC=.06, SOPE=10., KSUB=10., RDMSOL=120.)
    site = dict(WOFOST72SiteDataProvider(WAV=20.))
    params = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    first, last = expected_days[0], expected_days[-1]
    management = [{first: {"CropCalendar": dict(crop_name=config.crop_name, variety_name=config.variety,
                    crop_start_date=first, crop_start_type=config.start_type, crop_end_date=last,
                    # PCSE counts sowing/emergence day as duration zero.
                    crop_end_type=config.end_type, max_duration=(last - first).days),
                    "TimedEvents": None, "StateEvents": None}}]
    provider = _weather_provider(frame, config, frame.IRRAD if radiation is None else radiation)
    model = Wofost72_PP(params, provider, management)
    model.run_till_terminate()
    daily = pd.DataFrame(model.get_output())
    summaries = model.get_summary_output()
    if not summaries:
        raise RuntimeError("模型没有生成季节汇总，不能输出最终产量。")
    summary = summaries[0]
    if config.end_type == "harvest":
        # Harvested on the calendar date: the season is complete when the simulation reached that day
        # (PCSE 6.0.13 leaves DOH empty in the summary of a calendar harvest).
        mature = pd.to_datetime(daily.day.iloc[-1]).date() == last
    else:
        mature = summary.get("DOM") is not None and float(summary.get("DVS", 0)) >= 1.999
    for key in ("TWSO", "TAGP", "DVS", "CTRAT"):
        if not math.isfinite(float(summary[key])) or summary[key] < 0:
            raise ValueError(f"模型输出 {key} 无效。")
    if frame.RAIN_mm_day.isna().any():
        # PP yield can still be tested, but WaterbalancePP uses rainfall in EVS.
        # Do not publish a zero-rain artefact as measured/valid soil evaporation.
        summary["CEVST"] = None
    actual_weather = pd.DataFrame([{key: getattr(provider(day), key) for key in
        ("DAY", "IRRAD", "TMIN", "TMAX", "TEMP", "VAP", "WIND", "RAIN", "E0", "ES0", "ET0", "LAT", "LON", "ELEV")}
        for day in daily.day])
    return dict(daily=daily, weather=actual_weather, summary=summary, mature=mature, parameters=parameter_info,
                soil=soil, site=site, management=management)


def validate_radiation(frame, radiation, scenario_id, grid):
    required = {"day", "scenario_id", "point_id", "x_m", "y_m", "cell_area_m2", "total_Wh_m2_day"}
    if not required.issubset(radiation.columns):
        raise ValueError(f"辐射表缺列：{sorted(required - set(radiation.columns))}；整期 Wh 汇总不能导入。")
    selected = radiation[radiation.scenario_id == scenario_id].copy()
    if selected.empty:
        raise ValueError(f"辐射表没有情景 {scenario_id}。")
    selected["day"] = pd.to_datetime(selected.day, errors="raise").dt.date
    if selected.duplicated(["day", "point_id"]).any():
        raise ValueError("同一天同一个采样点有重复辐射记录。")
    expected = grid.points()
    if set(selected.point_id) != set(expected.point_id):
        raise ValueError("采样点与统一研究地块不一致。")
    expected_days = set(frame.day)
    for pid, group in selected.groupby("point_id"):
        if set(group.day) != expected_days:
            raise ValueError(f"点 {pid} 未完整覆盖作物配置日期；不允许将 30 天外推成整个产季。")
        point = expected.loc[expected.point_id == pid].iloc[0]
        for key in ("x_m", "y_m", "cell_area_m2"):
            if not np.allclose(pd.to_numeric(group[key]), point[key], rtol=0, atol=1e-9):
                raise ValueError("同一个点的位置或代表面积发生变化，不能连续模拟。")
    vals = selected[["total_Wh_m2_day", "cell_area_m2", "x_m", "y_m"]].to_numpy(float)
    if not np.isfinite(vals).all() or (selected.total_Wh_m2_day < 0).any():
        raise ValueError("辐射值必须为非负有限数；不能把缺失值补成 0。")
    if "direct_Wh_m2_day" in selected and "diffuse_residual_Wh_m2_day" in selected:
        components = selected[["direct_Wh_m2_day", "diffuse_residual_Wh_m2_day"]].to_numpy(float)
        if not np.isfinite(components).all() or (components < 0).any():
            raise ValueError("直射/散射残差出现负值或缺失，请检查 Radiance 精度，不能悄悄截断。")
        if not np.allclose(components.sum(axis=1), selected.total_Wh_m2_day, atol=1e-6, rtol=1e-6):
            raise ValueError("直射加散射残差与总辐射不一致。")
    return selected.sort_values(["point_id", "day"])


@dataclass(frozen=True)
class GroundGrid:
    # Common WORLD coordinates for all four PV scenarios; this is an explicit example plot.
    x_min: float = -16.0
    x_max: float = 16.0
    y_min: float = -16.0
    y_max: float = 16.0
    nx: int = 15
    ny: int = 15
    z_m: float = .05

    def points(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("地块参数必须为有限数。")
        if self.x_max <= self.x_min or self.y_max <= self.y_min or self.z_m <= 0:
            raise ValueError("地块范围/传感器高度无效。")
        if not isinstance(self.nx, int) or not isinstance(self.ny, int) or min(self.nx, self.ny) < 1:
            raise ValueError("网格点数必须为正整数。")
        dx, dy = (self.x_max - self.x_min) / self.nx, (self.y_max - self.y_min) / self.ny
        # Matches AnalysisObj._linePtsMakeDict order: outer Nx, inner Ny (Nz=1).
        return pd.DataFrame([dict(point_id=f"p{i:03d}_{j:03d}", x_m=self.x_min + (i + .5) * dx,
                                  y_m=self.y_min + (j + .5) * dy, cell_area_m2=dx * dy)
                             for i in range(self.nx) for j in range(self.ny)])

    def scan(self):
        points = self.points()
        dx, dy = (self.x_max - self.x_min) / self.nx, (self.y_max - self.y_min) / self.ny
        return dict(xstart=float(points.iloc[0].x_m), ystart=float(points.iloc[0].y_m), zstart=self.z_m,
                    sx_xinc=dx, sx_yinc=0., sx_zinc=0., xinc=0., yinc=dy, zinc=0.,
                    Nx=self.nx, Ny=self.ny, Nz=1, orient="0 0 -1"), dx * dy


def evaluate_crop_yield_for_hao(frame, config, grid, radiation=None, *, scenario_id="natural_light",
                                provenance=None, spatial_mode="pointwise", baseline=None):
    """Hao's crop interface, not his PV/economic optimizer. Returns summary + run folder."""
    if spatial_mode not in ("pointwise", "area_mean_approximation"):
        raise ValueError("spatial_mode 只能为 pointwise 或 area_mean_approximation。")
    points = grid.points()
    area = float(points.cell_area_m2.sum())
    natural = scenario_id == "natural_light"
    if natural and radiation is not None:
        raise ValueError("自然光照项目不能混入遮阴辐射表。")
    if not natural and (radiation is None or not provenance):
        raise ValueError("APV 情景必须提供完整逐日辐射及来源记录。")
    if natural:
        jobs = [("whole_plot", area, frame.IRRAD.to_numpy())]
    else:
        selected = validate_radiation(frame, radiation, scenario_id, grid)
        if spatial_mode == "area_mean_approximation":
            selected["weighted"] = selected.total_Wh_m2_day * selected.cell_area_m2
            r = selected.groupby("day").weighted.sum().reindex(frame.day).to_numpy() / area
            jobs = [("area_mean", area, r * 3600)]
        else:
            jobs = [(pid, float(g.cell_area_m2.iloc[0]),
                     g.set_index("day").reindex(frame.day).total_Wh_m2_day.to_numpy(float) * 3600)
                    for pid, g in selected.groupby("point_id", sort=True)]
    point_summaries, daily_frames, weather_frames = [], [], []
    model_result = None
    for pid, weight, irrad in jobs:
        model_result = run_crop(frame, config, irrad)
        s = model_result["summary"]
        point_summaries.append(dict(point_id=pid, cell_area_m2=weight, mature=model_result["mature"],
                                    maturity_date=str(s["DOM"]) if s.get("DOM") else None,
                                    TWSO_kg_DM_ha=float(s["TWSO"]), TAGP_kg_DM_ha=float(s["TAGP"]),
                                    DVS=float(s["DVS"]), transpiration_mm=float(s["CTRAT"]) * 10))
        d = model_result["daily"].copy()
        d.insert(0, "point_id", pid)
        daily_frames.append(d)
        w = model_result["weather"].copy()
        w.insert(0, "point_id", pid)
        weather_frames.append(w)
    point_table = pd.DataFrame(point_summaries)
    all_mature = bool(point_table.mature.all())
    weighted_yield = float(np.average(point_table.TWSO_kg_DM_ha, weights=point_table.cell_area_m2))
    comparison = fingerprint(dict(config=asdict(config), weather=frame.to_json(date_format="iso"),
                                   grid=asdict(grid), params=model_result["parameters"],
                                   bridge_sha256=sha256(__file__), pcse=version("pcse"), spatial_mode=spatial_mode))
    payload = dict(schema_version="hao-crop-yield-v1", scenario_id=scenario_id,
        model="Wofost72_PP", pcse_version=version("pcse"), adapter_version=ADAPTER_VERSION,
        status="mature_uncalibrated_model_result" if all_mature else "incomplete_season_do_not_optimise",
        field_validated=False, all_points_mature=all_mature, crop_variety=config.variety,
        sowing_or_emergence_date=config.start, simulation_end=config.end,
        common_planted_area_m2=area, spatial_mode="uniform_natural" if natural else spatial_mode,
        yield_kg_DM_ha=weighted_yield if all_mature else None,
        yield_kg_DM_m2=weighted_yield / 10000 if all_mature else None,
        total_grain_DM_kg_on_plot=weighted_yield / 10000 * area if all_mature else None,
        interim_storage_organ_DM_kg_ha=weighted_yield,
        aboveground_biomass_kg_DM_ha=float(np.average(point_table.TAGP_kg_DM_ha, weights=point_table.cell_area_m2)),
        model_transpiration_mm=float(np.average(point_table.transpiration_mm, weights=point_table.cell_area_m2)),
        soil_evaporation_mm=None, irrigation_requirement_mm=None,
        water_mode="PP: no water limitation; no 80% transpiration control",
        relative_yield_to_natural=None, comparison_fingerprint=comparison,
        input_provenance=provenance or {"source": "EPW natural GHI"},
        limitations=["Example cultivar/calendar/soil are not Aachen field calibration.",
                     "Tianze near-ground shortwave used as effective crop incident IRRAD: first-order coupling.",
                     "WOFOST72 reconstructs diffuse fraction internally; supplied direct/diffuse are audit-only.",
                     "No hourly shade sequence, spectral response, canopy-height feedback or APV microclimate.",
                     "ET recomputed with shaded IRRAD using open-field formula: approximate, not measured APV ET.",
                     "No irrigation optimisation, LER, revenue or PV electricity calculation.",
                     "All selected grid cells assumed planted; supports/paths excluded only by future explicit mask."])
    if baseline is not None:
        if baseline.get("scenario_id") != "natural_light" or baseline.get("comparison_fingerprint") != comparison:
            raise ValueError("自然对照必须采用完全相同的天气、作物参数、日历、地块和空间求解设置。")
        by = baseline.get("yield_kg_DM_ha")
        if all_mature and by is not None and by > 0:
            payload["relative_yield_to_natural"] = weighted_yield / by
    folder = new_run("aachen_research")
    payload["run_id"] = folder.name
    point_table.to_csv(folder / "point_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(folder / "crop_daily.csv", index=False, encoding="utf-8-sig")
    pd.concat(weather_frames, ignore_index=True).to_csv(folder / "actual_model_weather.csv", index=False, encoding="utf-8-sig")
    save_json(folder / "units.json", dict(IRRAD="J/m2/day", E0="cm/day", ES0="cm/day", ET0="cm/day",
        RAIN="cm/day; zero may be an explicitly accepted PP placeholder, see config", VAP="hPa", WIND="m/s at 2m",
        TWSO="kg dry matter/ha", TAGP="kg dry matter/ha", LAI="m2/m2", TRA="cm/day",
        total_Wh_m2_day="Wh/m2/day", cell_area_m2="m2"))
    frame.to_csv(folder / "base_weather_daily.csv", index=False, encoding="utf-8-sig")
    if not natural:
        selected.to_csv(folder / "radiation_daily_points.csv", index=False, encoding="utf-8-sig")
    save_json(folder / "config_and_parameters.json", dict(config=asdict(config), grid=asdict(grid),
                    parameters=model_result["parameters"], soil=model_result["soil"], site=model_result["site"],
                    management=model_result["management"]))
    save_json(folder / "Hao_crop_yield.json", payload)
    pd.DataFrame([{k: v for k, v in payload.items() if not isinstance(v, (dict, list))}]).to_csv(
        folder / "Hao_crop_yield.csv", index=False, encoding="utf-8-sig")
    return payload, folder
