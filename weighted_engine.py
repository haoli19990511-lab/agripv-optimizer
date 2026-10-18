"""Evaluation engine: physics (Radiance via the notebook functions), crop (WOFOST), economics,
NSGA-II search, re-check, and the resumable physics and evaluation caches."""
from dataclasses import dataclass, asdict, replace, fields
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import os
import shutil
import time
import numpy as np
import pandas as pd
import bridge
from sampling_core import (SamplingConfig, make_sampling_plan, design_grid, fingerprint, compare_sampling,
                           resolve_array, field_length_m)


@dataclass(frozen=True)
class EngineConfig:
    # Defaults used in the thesis; sources in the comments.
    grain_dry_matter_fraction: float = .86
    conventional_peak_w: float = 590.  # LONGi LR5-72HGD datasheet
    # semi-transparent front efficiency is on ACTIVE CELL AREA;
    # conventional efficiency is derived on the WHOLE MODULE envelope.
    semi_cell_efficiency: float = .228  # datasheet module efficiency, applied to the cell area
    wiring_inverter_factor: float = .96
    temperature_factor: float = .97  # explicit constant approximation
    module_semi_eur_kwp: float = 650.
    module_conv_eur_kwp: float = 220.
    # Structure: 300 EUR/kWp up to 3 m, +150 EUR/kWp per m above -> 600 EUR/kWp at 5 m
    # (SYMBIOSYST D7.3 p.13: ~300 for horticulture < 5 m, ~600 for arable crops > 5 m).
    structure_base_eur_kwp: float = 300.
    structure_reference_height_m: float = 3.
    structure_extra_eur_kwp_m: float = 150.
    # Single-axis tracker premium: SYMBIOSYST D7.3 p.13, ~650 EUR/kWp fixed vs ~800 EUR/kWp tracked PV.
    tracker_extra_eur_kwp: float = 150.
    # Inverters, electrical BOS, installation, fees and site preparation: Bellone et al. 2024 Table 10
    # (~500 EUR/kWp) + D7.3 site preparation (150 EUR/kWp). Total CAPEX of the fixed
    # elevated system is then 1170-1470 EUR/kWp, inside D7.3's 1500 +/- 425 EUR/kWp for arable agri-PV.
    other_capex_eur_kwp: float = 650.
    # Vertical bifacial fences: low fixed structure, SYMBIOSYST D7.3 p.13 (~50 EUR/kWp, low fixed racks);
    # replaces structure_base/extra for vertical designs. Strip under each vertical row that cannot be
    # cropped (posts, machinery distance), m per row (assumption).
    vertical_structure_eur_kwp: float = 50.
    # German agri-PV bonus (EEG 2023 Sec. 38b, "hoch aufgestaenderte" agri-PV, >= 2.1 m clear height):
    # 0.5 ct/kWh on top of the market revenue for awards from 2025 on (1.2 ct/kWh in 2023), only for
    # tendered plants (Energieagentur RLP 2026). Report the results with and without it.
    # Not granted to vertical systems. 0 switches it off.
    agripv_bonus_eur_kwh: float = .005
    agripv_bonus_min_clearance_m: float = 2.1
    vertical_row_strip_m: float = .5
    opex_fraction: float = .01
    crop_price_eur_t: float = 213.1           # winter wheat, NRW planning data 2024 (see crops.py)
    crop_cost_eur_planted_ha: float = 1120.
    lifetime_years: int = 25
    discount_rate: float = .05
    planted_fraction: float = .97  # elevated systems: area lost to the support posts (assumption)
    minimum_crop_retention: float = .66  # crop yield >= 66 % of the open field (DIN SPEC 91434)
    minimum_clearance_m: float = 2.5
    minimum_clear_corridor_m: float = 1.5
    trace_ground_components: bool = False  # direct/diffuse split is validation-only; costs one extra trace per sky


def load_price_profile(hours, directory, mode, flat_rate, years):
    idx = pd.DatetimeIndex(hours)
    if mode == "flat_assumption":
        if not np.isfinite(flat_rate):
            raise ValueError("Invalid assumed price")
        return pd.Series(float(flat_rate), index=idx)
    if mode != "csv" or not years:
        raise ValueError("Price mode must be flat_assumption or csv; CSV mode requires years")
    frames = []
    for year in years:
        path = Path(directory)/f"electricity_price_{year}_DE_LU.csv"
        df = pd.read_csv(path)
        expected = 8784 if pd.Timestamp(f"{year}-01-01").is_leap_year else 8760
        if set(["hour", "price_eur_per_kwh"])-set(df) or len(df) != expected:
            raise ValueError(f"{path}: requires complete UTC year, hour and price_eur_per_kwh")
        h = pd.to_numeric(df.hour, errors="raise").to_numpy()
        if not np.array_equal(h, np.arange(expected)):
            raise ValueError(f"{path}: hour must be exact ordered 0..{expected-1}, no duplicates/gaps")
        prices = pd.to_numeric(df.price_eur_per_kwh, errors="raise").to_numpy(float)
        if not np.isfinite(prices).all():
            raise ValueError(f"{path}: nonfinite prices")
        # Prices are UTC, target EPW timezone is fixed standard time (no DST).
        stamps = pd.date_range(f"{year}-01-01", periods=expected, freq="h", tz="UTC").tz_convert(idx.tz)
        mask = ~((stamps.month == 2)&(stamps.day == 29))
        keys = stamps[mask].strftime("%m-%d-%H")
        if not keys.is_unique:
            raise ValueError("Price month/day/hour keys are not unique")
        frames.append(pd.Series(prices[mask], index=keys))
    combined = pd.concat(frames, axis=1)
    if len(combined) != 8760 or combined.isna().any().any():
        raise ValueError("All requested price years must cover the same 8760 standard-time keys")
    typical = combined.mean(axis=1)
    values = typical.reindex(idx.strftime("%m-%d-%H"))
    if values.isna().any():
        raise ValueError("Missing target-hour prices (leap target calendar unsupported)")
    return pd.Series(values.to_numpy(), index=idx)


DESIGN_CLASSES = {"fixed": "PVDesign", "single_axis": "TrackingDesign", "vertical": "VerticalPVDesign"}


def is_tracking(design):
    return hasattr(design, "axis_azimuth")


def mounting_of(design):
    if is_tracking(design):
        return "single_axis"
    if type(design).__name__ == "VerticalPVDesign":
        return "vertical"
    return "fixed"


def build_design(cls, mapping):
    """Design object from a dict/row; only constructor fields (vertical tilt is fixed at 90)."""
    from dataclasses import fields as _fields
    return cls(**{f.name: (bool(mapping[f.name]) if f.name == "backtrack" else float(mapping[f.name]))
                  for f in _fields(cls) if f.init})


def mounting_geometry(design, module):
    """Height for the structure cost, lowest module edge and clear corridor between rows.

    Fixed tilt: the clearance is the lower module edge. Single axis: the structure height is
    the hub (torque-tube) height; clearance and corridor are taken with the modules turned to
    the rotation limit, the position in which machinery passes and the lower edge is lowest."""
    if mounting_of(design) == "vertical":
        # No overhead clearance needed: machinery drives between the vertical rows.
        return dict(mounting="vertical", structure_height_m=float(design.clearance_height + module.module_y),
                    min_clearance_m=float(design.clearance_height), clearance_applies=False,
                    clear_corridor_m=float(design.pitch-module.module_thickness))
    if is_tracking(design):
        angle = min(design.limit_angle, np.degrees(np.arctan2(module.module_y, module.module_thickness)))
        a = np.radians(angle)
        low = design.hub_height - .5*(module.module_y*np.sin(a) + module.module_thickness*np.cos(a))
        return dict(mounting="single_axis", structure_height_m=float(design.hub_height),
                    min_clearance_m=float(low), clearance_applies=True,
                    clear_corridor_m=float(design.pitch-module.module_y*np.cos(np.radians(design.limit_angle))))
    return dict(mounting="fixed", structure_height_m=float(design.clearance_height),
                min_clearance_m=float(design.clearance_height), clearance_applies=True,
                clear_corridor_m=float(design.pitch-module.module_y*np.cos(np.radians(design.tilt))))


def economic_metrics(kind, hourly, prices, design, module, array, grid,
                     crop_yield, natural_yield, mature, pv_reference, config):
    """Annual PV revenue plus exactly ONE crop harvest; all land KPIs use grid area."""
    area_ha = grid.area_m2/10000.
    n = array.n_mods*array.n_rows
    active = module.numcellsx*module.numcellsy*module.xcell*module.ycell
    envelope = module.module_x*module.module_y
    efficiency = config.semi_cell_efficiency if kind == "semi_transparent" else config.conventional_peak_w/(1000*envelope)
    peak_w = active*1000*efficiency if kind == "semi_transparent" else config.conventional_peak_w
    incident = hourly[f"{kind}_array_front_effective_incident_energy_Wh_hour"].to_numpy(float, copy=True)
    incident += hourly[f"{kind}_array_rear_effective_incident_energy_Wh_hour"].to_numpy(float)
    if not np.isfinite(incident).all() or (incident < 0).any():
        raise ValueError("Invalid effective incident energy")
    kwh = incident*efficiency*config.temperature_factor*config.wiring_inverter_factor/1000.
    if len(kwh) != len(prices):
        raise ValueError("Annual electricity and price hours must agree")
    electrical_revenue = float(np.dot(kwh, np.asarray(prices, float)))
    kwp = n*peak_w/1000.
    geometry = mounting_geometry(design, module)
    bonus_eligible = (geometry["mounting"] != "vertical"
                      and geometry["min_clearance_m"] >= config.agripv_bonus_min_clearance_m)
    agripv_bonus = config.agripv_bonus_eur_kwh*float(kwh.sum()) if bonus_eligible else 0.
    structure_rate = config.structure_base_eur_kwp + config.structure_extra_eur_kwp_m*max(
        0., geometry["structure_height_m"]-config.structure_reference_height_m)
    if geometry["mounting"] == "single_axis":
        structure_rate += config.tracker_extra_eur_kwp
    elif geometry["mounting"] == "vertical":
        structure_rate = config.vertical_structure_eur_kwp
    module_rate = config.module_semi_eur_kwp if kind == "semi_transparent" else config.module_conv_eur_kwp
    capex = kwp*(module_rate+structure_rate+config.other_capex_eur_kwp)
    opex = capex*config.opex_fraction
    crop_yield = float(crop_yield) if crop_yield is not None else 0.
    planted = config.planted_fraction
    if geometry["mounting"] == "vertical":
        # One uncropped strip per vertical row instead of the uniform loss to elevated posts.
        planted = max(0., 1. - array.n_rows*config.vertical_row_strip_m/field_length_m(array, design))
    crop_revenue = crop_yield*area_ha*planted*config.crop_price_eur_t
    crop_cost = area_ha*planted*config.crop_cost_eur_planted_ha
    cashflow = electrical_revenue+agripv_bonus+crop_revenue-opex-crop_cost
    if config.lifetime_years < 1 or config.discount_rate < 0:
        raise ValueError("Lifetime must be positive, discount rate nonnegative")
    r, years = config.discount_rate, config.lifetime_years
    crf = 1/years if r == 0 else r/(1-(1+r)**(-years))
    annualised_capex = capex*crf
    annual_net = cashflow-annualised_capex
    if not np.isfinite(natural_yield) or natural_yield <= 0 or pv_reference <= 0:
        raise ValueError("Reference crop and PV yields must be positive")
    crop_ratio = crop_yield/natural_yield*planted
    pv_ratio = float(kwh.sum())/area_ha/pv_reference
    corridor = geometry["clear_corridor_m"]
    clearance = geometry["min_clearance_m"]
    reasons = []
    if not mature: reasons.append("crop_not_mature")
    if crop_ratio < config.minimum_crop_retention: reasons.append("crop_retention_below_threshold")
    clearance_gap = max(0., config.minimum_clearance_m-clearance) if geometry["clearance_applies"] else 0.
    if clearance_gap > 0: reasons.append("insufficient_clearance")
    if corridor < config.minimum_clear_corridor_m: reasons.append("insufficient_clear_corridor")
    # Continuous measure for NSGA-II: 0 when feasible, else the summed relative shortfall.
    # (A 0/1 flag gives the optimiser no direction while the whole population is infeasible.)
    violation = (max(0., config.minimum_crop_retention-crop_ratio)/config.minimum_crop_retention
                 + clearance_gap/config.minimum_clearance_m
                 + max(0., config.minimum_clear_corridor_m-corridor)/config.minimum_clear_corridor_m
                 + (0. if mature else 1.))
    return dict(pv_type=kind, footprint_ha=area_ha, constraint_violation=float(violation), planted_area_ha=area_ha*planted,
        crop_yield_t_ha=crop_yield, natural_yield_t_ha=natural_yield,
        crop_all_points_mature=bool(mature), LER_crop=crop_ratio, LER_pv=pv_ratio, LER=crop_ratio+pv_ratio,
        electrical_kwh=float(kwh.sum()), electrical_kwh_per_ha=float(kwh.sum())/area_ha,
        electrical_revenue_eur=electrical_revenue, crop_revenue_eur=crop_revenue,
        crop_cost_eur=crop_cost, annual_opex_eur=opex, agripv_bonus_eur=agripv_bonus,
        revenue_eur=electrical_revenue+agripv_bonus+crop_revenue,
        capture_price_eur_kwh=electrical_revenue/float(kwh.sum()) if kwh.sum() > 0 else float("nan"),
        annual_operating_cashflow_eur=cashflow, annualised_capex_eur=annualised_capex,
        annual_net_eur=annual_net, annual_net_eur_ha=annual_net/area_ha,
        installed_kwp=kwp, capex_total_eur=capex, module_peak_w=peak_w,
        payback_period_years=capex/cashflow if cashflow > 0 else None,
        feasible=not reasons, infeasible_reason=";".join(reasons), clear_corridor_m=float(corridor),
        mounting=geometry["mounting"], structure_height_m=geometry["structure_height_m"],
        min_clearance_m=float(clearance))


def mark_pareto(table):
    """Add a boolean 'pareto' column: non-dominated feasible rows per construction
    (maximise LER and annual net income per hectare). Returns the same frame."""
    table["pareto"] = False
    for kind, group in table.groupby("pv_type"):
        feasible = group["feasible"].astype(bool).to_numpy()
        values = group[["LER", "annual_net_eur_ha"]].to_numpy(float)
        for local_i, idx in enumerate(group.index):
            if not feasible[local_i]:
                continue
            dominates = feasible & np.all(values >= values[local_i], axis=1) & np.any(values > values[local_i], axis=1)
            if not dominates.any():
                table.loc[idx, "pareto"] = True
    return table


def _code_digest(code):
    """Hash bytecode, names and constants only -- NOT file name or line numbers.

    marshal.dumps(code) embeds co_filename, which differs between a notebook
    kernel and a worker process executing the same cells, so the two processes
    would otherwise compute different cache keys for identical code.
    """
    h = hashlib.sha256()
    h.update(code.co_code)
    h.update(repr((code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)).encode())
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            h.update(_code_digest(const).encode())
        elif isinstance(const, frozenset):
            # repr order of a frozenset depends on PYTHONHASHSEED, which differs per process
            h.update(repr(sorted(map(repr, const))).encode())
        else:
            h.update(repr(const).encode())
    return h.hexdigest()


def _kill_tree(proc):
    """Stop a worker AND the Radiance processes it started (taskkill /T on Windows)."""
    import subprocess
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()


class WeightedEngine:
    def __init__(self, namespace, crop_config, config, sampling, reference_design,
                 price_mode="flat_assumption", flat_price=.08, price_years=(),
                 parallel_designs=1, workers_per_design=None, notebook_path=None):
        self.ns, self.crop_config, self.config, self.sampling = namespace, crop_config, config, sampling
        if isinstance(parallel_designs, bool) or not isinstance(parallel_designs, int) or parallel_designs < 1:
            raise ValueError("parallel_designs must be a positive integer")
        self.parallel_designs, self.workers_per_design, self.notebook_path = parallel_designs, workers_per_design, notebook_path
        self._price_args = (price_mode, flat_price, list(price_years))
        self.scenario, self.module, self.array = (namespace[x] for x in ("scenario", "module_specification", "array_specification"))
        self.reference_design = reference_design
        self.root = Path(namespace["PROJECT_ROOT"])
        self.cache = self.root/"results"/"cache_v5"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.crop_days = crop_config.dates()
        first, last = namespace["cultivation_period_bounds"](self.scenario)
        if (last-first).days+1 != 365 or any(d.month == 2 and d.day == 29 for d in pd.date_range(first,last)):
            raise ValueError("Use exactly 365 non-leap calendar days for annual PV")
        # Ground irradiation is traced for EVERY day of the PV year, independent of the crop, so that the
        # crop can be changed without new ray tracing. A crop day outside the PV year (e.g. maize harvested
        # in October) uses the day of the PV year with the same month and day: the weather is a typical year.
        self.pv_days = list(pd.date_range(first, last).date)
        by_month_day = {(d.month, d.day): d for d in self.pv_days}
        self._pv_day = {d: by_month_day[(d.month, d.day)] for d in self.crop_days}
        if len(set(self._pv_day.values())) != len(self.crop_days):
            raise ValueError("A crop season may not be longer than one year")
        if not 0 < config.planted_fraction <= 1 or not 0 < config.grain_dry_matter_fraction <= 1:
            raise ValueError("Invalid area/dry-matter fraction")
        for field in ("minimum_crop_retention", "temperature_factor", "wiring_inverter_factor"):
            if not 0 <= getattr(config, field) <= 1: raise ValueError(f"Invalid {field}")
        # Read weather without tracing or constructing module scenes.
        cwd = Path.cwd()
        with TemporaryDirectory(prefix="agripv_weather_") as tmp:
            try:
                demo = namespace["br"].RadianceObj(name="weather_only", path=tmp, hpc=True)
                hours, _, _, _ = namespace["read_hourly_cultivation_weather"](demo, self.scenario)
                self.hourly_weather = namespace["ALIGNED_HOURLY_WEATHER"].copy()
            finally:
                os.chdir(cwd)
        self.hours = pd.DatetimeIndex(hours)
        self.prices = load_price_profile(hours, self.root/"prices", price_mode, flat_price, price_years)
        self.price_assumptions = dict(mode=price_mode, flat_eur_kwh=flat_price if price_mode == "flat_assumption" else None,
                                      years=list(price_years) if price_mode == "csv" else [])
        self.weather, self.weather_metadata = self._crop_weather()
        functions = ["read_hourly_cultivation_weather", "make_hourly_sky", "create_selected_module",
                     "analyze_array", "analyze_module", "analyze_ground_hour", "trace_direct_ground_irradiation",
                     "calculate_hourly_results", "make_module_surface_scans", "make_rectangular_surface_scans",
                     "module_geometry", "glass_margin_text", "add_hazy_glass_material"]
        # Fixed-tilt runs digest exactly the v6.5 list, so their cached ray tracing stays valid.
        # Single-axis runs (MOUNTING = "single_axis" in Section C) add the tracking functions.
        self.mounting = namespace.get("MOUNTING", "fixed")
        if self.mounting not in DESIGN_CLASSES:
            raise ValueError(f"MOUNTING must be one of {list(DESIGN_CLASSES)}")
        if self.mounting == "single_axis":
            functions += ["calculate_hourly_results_tracking", "analyze_tracking_array", "analyze_tracking_module",
                          "make_tracking_surface_scans", "make_tracking_rectangular_scans",
                          "validate_tracking_design", "tracking_min_clearance"]
        if self.mounting == "vertical":
            functions += ["calculate_hourly_results_vertical", "analyze_vertical_array", "analyze_vertical_module",
                          "make_vertical_surface_scans", "make_vertical_rectangular_scans", "validate_vertical_design"]
        settings = {k: v for k, v in asdict(namespace["SETTINGS"]).items() if k != "n_workers"}
        digests = {f:_code_digest(namespace[f].__code__) for f in functions}
        files = {p:bridge.sha256(Path(__file__).parent/p) for p in ("bridge.py", "sampling_core.py", "weighted_engine.py")}
        packages = {p:bridge.version(p) for p in ("pcse", "pvlib", "bifacial-radiance", "numpy", "pandas")}
        # Only the Python code that shapes the Radiance run (sampling plan, plot grids, row count,
        # the simulation call) is digested for the physics key, so edits to economics, export,
        # re-check or README tooling do NOT invalidate hours of cached ray tracing.
        import sampling_core
        engine_code = {
            "make_sampling_plan": sampling_core.make_sampling_plan, "design_grid": sampling_core.design_grid,
            "resolve_array": sampling_core.resolve_array, "field_length_m": sampling_core.field_length_m,
            "PlotGrid.points": sampling_core.PlotGrid.points, "PlotGrid.scan": sampling_core.PlotGrid.scan,
            "PlotGrid.area_m2": sampling_core.PlotGrid.area_m2.fget,
            "WeightedEngine._simulation": WeightedEngine._simulation, "WeightedEngine._grids": WeightedEngine._grids}
        engine_digests = {name: _code_digest(fn.__code__) for name, fn in engine_code.items()}
        # Physics signature: everything that changes the Radiance result. Prices, costs and
        # the reference design are NOT in it, so changing them reuses the cached ray tracing.
        # The crop is NOT part of it: the ground is traced for the whole PV year (pv_days).
        self.physics_signature = dict(version="nsga2-v7-fullyear-ground", scenario=asdict(self.scenario),
            module=asdict(self.module), array=asdict(self.array), settings=settings, ground_days="pv_year",
            trace_ground_components=config.trace_ground_components,
            weather_sha256=bridge.sha256(self.root/"weather"/self.scenario.weather_filename),
            functions=digests, engine_code=engine_digests, packages=packages)
        # Full signature (evaluation cache): physics plus prices, economic/crop parameters,
        # the reference design and the hashes of the whole source files.
        self.signature = dict(self.physics_signature, config=asdict(config), files=files, crop=asdict(crop_config),
            crop_parameters_sha256=bridge.sha256(bridge.crop_file_path(crop_config)),
            prices=self.prices.tolist(), reference_design=asdict(reference_design))
        self._reference = None
        self.constructions = ("semi_transparent", "conventional")  # ConstructionEngine narrows this
        plan = make_sampling_plan(hours, self.hourly_weather.GHI, self.hourly_weather.DHI, sampling)
        print(f"Sampling {sampling.mode}: {plan.sample_hour.nunique()} skies/type/design; full: {(self.hourly_weather.GHI>0).sum()}")
        print("Prices:", self.price_assumptions)

    def _crop_weather(self):
        # Reaggregate ALL meteorology using exactly the same clock as Radiance.
        frame, metadata = bridge.read_epw_daily(self.root/"weather"/self.scenario.weather_filename, self.crop_config)
        h = self.hourly_weather.copy()
        h.index = self.hours
        if ((h._crop_temp >= 99.9)|(h._crop_dew >= 99.9)|(h._crop_wind >= 999)|(h._crop_wind < 0)).any():
            raise ValueError("EPW contains invalid crop weather")
        h["vap"] = 6.108*np.exp(17.27*h._crop_dew/(h._crop_dew+237.3))
        rows = {}
        for day, group in h.groupby(h.index.date):
            if len(group) != 24: raise ValueError("Incomplete crop day")
            rain_ok = ((group._crop_rain >= 0)&(group._crop_rain < 999)&(group._crop_rain_period == 1)).all()
            rows[day] = dict(TMIN=float(group._crop_temp.min()), TMAX=float(group._crop_temp.max()),
                TEMP=float(group._crop_temp.mean()), WIND=float(group._crop_wind.mean()), VAP=float(group.vap.mean()),
                IRRAD=float(group.GHI.sum())*3600., DHI_Wh_m2_day=float(group.DHI.sum()),
                RAIN_mm_day=float(group._crop_rain.sum()) if rain_ok else None)
        for col in next(iter(rows.values())):
            frame[col] = [rows[self._pv_day[d]][col] for d in frame.day]
        metadata["shared_clock"] = self.scenario.weather_time_basis
        metadata["natural_irrad_source"] = "same DNI/DHI and actual solar geometry as Radiance"
        return frame, metadata

    def _grids(self, design):
        return {kind:design_grid(design, self.module, self.array, kind,
                                self.scenario.ground_grid_x, self.scenario.ground_grid_y,
                                self.ns["SETTINGS"].ground_sensor_height)
                for kind in ("semi_transparent", "conventional")}

    def _physics_key(self, design, sampling, crop=True):
        return fingerprint(dict(signature=self.physics_signature, design=asdict(design), sampling=asdict(sampling), crop=crop))

    def _physics_cached(self, design, sampling, crop=True):
        return (self.cache/self._physics_key(design, sampling, crop)/"manifest.json").exists()

    def _evaluation_cached(self, design, sampling, spatial_mode="area_mean_approximation"):
        key = fingerprint(dict(signature=self.signature, design=asdict(design), sampling=asdict(sampling), spatial_mode=spatial_mode))
        return (self.cache/f"evaluation_{key}.json").exists()

    def _notebook_file(self):
        path = self.notebook_path or self.ns.get("NOTEBOOK_FILE")
        if not path:
            raise ValueError("Parallel evaluation needs the notebook file name: set NOTEBOOK_FILE in Section D")
        path = Path(path)
        if not path.is_absolute():
            path = Path(self.ns.get("WORKING_DIRECTORY", Path.cwd()))/path
        if not path.exists():
            raise FileNotFoundError(f"Notebook file not found: {path}. Save the notebook and check NOTEBOOK_FILE.")
        return path

    def precompute(self, designs, sampling=None):
        """Run the expensive Radiance physics for several designs at once.

        Each design runs in its OWN Python process (eval_worker.py), because the
        Radiance wrapper changes the process working directory and is therefore
        not thread-safe. Results travel through the disk cache only; afterwards
        evaluate() finds every design cached and just runs the cheap WOFOST part.
        With parallel_designs == 1 this does nothing (serial, in-process).
        """
        sampling = sampling or self.sampling
        if self.parallel_designs <= 1:
            return
        jobs, seen = [], set()
        if not self._physics_cached(self.reference_design, sampling, crop=False):
            jobs.append(("reference", self.reference_design))
        for design in designs:
            self.ns["validate_cell_array_fit"](design, self.module)
            key = self._physics_key(design, sampling, crop=True)
            if key in seen or self._physics_cached(design, sampling, crop=True) or self._evaluation_cached(design, sampling):
                continue
            seen.add(key)
            jobs.append(("design", design))
        if jobs:
            self._run_jobs(jobs, sampling)

    def _run_jobs(self, jobs, sampling):
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor
        notebook = self._notebook_file()
        jobdir = self.cache/"jobs"
        jobdir.mkdir(parents=True, exist_ok=True)
        workers = self.workers_per_design or max(2, (os.cpu_count() or 4)//self.parallel_designs)
        signature_sha = fingerprint(self.physics_signature)
        script = Path(__file__).with_name("eval_worker.py")
        cwd = str(Path(self.ns.get("WORKING_DIRECTORY", Path.cwd())))
        price_mode, flat_price, price_years = self._price_args
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        print(f"Parallel physics: {len(jobs)} job(s), {self.parallel_designs} at a time, "
              f"{workers} module threads each; logs in {jobdir}", flush=True)

        procs = []

        def run(job):
            kind, design = job
            key = self._physics_key(design, sampling, crop=(kind == "design"))[:12]
            spec = dict(kind=kind, design=asdict(design), sampling=asdict(sampling), crop=asdict(self.crop_config),
                        config=asdict(self.config), reference=asdict(self.reference_design), price_mode=price_mode,
                        flat_price=flat_price, price_years=price_years, signature_sha=signature_sha,
                        notebook=str(notebook), cwd=cwd, workers=workers,
                        constructions=list(self.constructions))
            spec_file, log_file = jobdir/f"{kind}_{key}.json", jobdir/f"{kind}_{key}.log"
            spec_file.write_text(json.dumps(spec, indent=1))
            t0 = time.time()
            print(f"  start {kind} {key}", flush=True)
            with open(log_file, "w", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, str(script), str(spec_file)], cwd=cwd, env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=(os.name != "nt"))
                procs.append(proc)
                code = proc.wait()
            print(f"  {'done' if code == 0 else 'FAILED'} {kind} {key} in {(time.time()-t0)/60:.1f} min", flush=True)
            return kind, key, code, log_file

        with ThreadPoolExecutor(max_workers=self.parallel_designs) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            try:
                results = [future.result() for future in futures]
            except BaseException:  # includes the notebook's Interrupt-kernel button
                for future in futures:
                    future.cancel()
                for proc in list(procs):
                    _kill_tree(proc)
                print("Interrupted: worker processes and their Radiance children were stopped.", flush=True)
                raise
        failed = [r for r in results if r[2] != 0]
        if failed:
            lines = []
            for kind, key, code, log_file in failed:
                tail = "\n".join(Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
                lines.append(f"[{kind} {key}] exit code {code}; log {log_file}:\n{tail}")
            raise RuntimeError("Parallel physics job(s) failed (finished jobs stay cached; rerun to resume):\n" + "\n\n".join(lines))

    def _simulation(self, design, sampling, crop=True):
        key = self._physics_key(design, sampling, crop)
        folder = self.cache/key
        manifest_path = folder/"manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if all((folder/p).exists() and bridge.sha256(folder/p) == sha for p,sha in manifest["files"].items()):
                hourly = pd.read_csv(folder/"hourly.csv.gz")
                grounds = {k:pd.read_csv(folder/f"ground_{k}.csv.gz") for k in self._grids(design)} if crop else {}
                print("Physics cache hit:", key[:12])
                return hourly, grounds, folder
            raise RuntimeError(f"Cache checksum mismatch: {folder}; remove this entry and rerun")
        grids = self._grids(design)
        t0 = time.time()
        hourly, grounds = self.ns["model_for_optimization"](
            design, self.scenario, self.module, resolve_array(self.array, design), calculate_linrui=crop, calculate_hao=True,
            ground_grid=grids, sampling_config=sampling, crop_dates=self.pv_days,
            constructions_to_run=None if crop else ("conventional",), persist_outputs=False,
            ground_components=self.config.trace_ground_components)
        if len(hourly) != 8760:
            raise RuntimeError("Annual hourly result must have 8760 rows")
        folder.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(folder/"hourly.csv.gz", index=False)
        for kind, ground in grounds.items(): ground.to_csv(folder/f"ground_{kind}.csv.gz", index=False)
        files = ["hourly.csv.gz"]+[f"ground_{k}.csv.gz" for k in grounds]
        manifest = dict(key=key, files={p:bridge.sha256(folder/p) for p in files}, elapsed_s=time.time()-t0,
                        sampling=asdict(sampling), design=asdict(design), grids={k:asdict(v) for k,v in grids.items()})
        bridge.save_json(manifest_path, manifest)
        return hourly, grounds, folder

    def _crop_ground(self, ground):
        """Daily ground irradiation of the crop season (from the full PV year), labelled with the crop dates."""
        g = ground.copy()
        g["day"] = pd.to_datetime(g.day).dt.date
        to_crop = {pv: crop for crop, pv in self._pv_day.items()}
        g = g[g.day.isin(to_crop)].copy()
        g["day"] = g.day.map(to_crop)
        if g.day.nunique() != len(self.crop_days):
            raise ValueError("Cached ground irradiation does not cover the crop season")
        return g.sort_values(["point_id", "day"]).reset_index(drop=True)

    def reference_yield(self, sampling):
        hourly, _, _ = self._simulation(self.reference_design, sampling, crop=False)
        area = self._grids(self.reference_design)["conventional"].area_m2/10000.
        envelope = self.module.module_x*self.module.module_y
        incident = sum(hourly[f"conventional_array_{face}_effective_incident_energy_Wh_hour"].to_numpy(float) for face in ("front", "rear"))
        energy = incident.sum()*self.config.conventional_peak_w/(1000*envelope)*self.config.temperature_factor*self.config.wiring_inverter_factor/1000
        if not np.isfinite(energy) or energy <= 0: raise ValueError("Reference PV has no positive energy")
        return float(energy/area)

    def evaluate(self, design, sampling=None, spatial_mode="area_mean_approximation"):
        sampling = sampling or self.sampling
        eval_key = fingerprint(dict(signature=self.signature, design=asdict(design), sampling=asdict(sampling), spatial_mode=spatial_mode))
        evaluation_file = self.cache/f"evaluation_{eval_key}.json"
        if evaluation_file.exists():
            return json.loads(evaluation_file.read_text())["rows"]
        self.ns["validate_cell_array_fit"](design, self.module)
        hourly, grounds, folder = self._simulation(design, sampling)
        actual = pd.DatetimeIndex(pd.to_datetime(hourly.hour_start))
        if not np.array_equal(actual.asi8, self.hours.asi8): raise ValueError("Cached hours do not match price/weather clock")
        prices = self.prices.to_numpy()
        pv_reference = self.reference_yield(sampling)
        rows = []
        for kind, grid in self._grids(design).items():
            # Same spatial_mode even for uniform natural; bridge fingerprint includes it.
            natural, _ = bridge.evaluate_crop_yield_for_hao(self.weather, self.crop_config, grid,
                scenario_id="natural_light", spatial_mode=spatial_mode)
            if not natural["all_points_mature"]: raise RuntimeError("Natural crop did not mature; fix crop calendar")
            radiation = self._crop_ground(grounds[kind])
            radiation["scenario_id"] = eval_key
            payload, _ = bridge.evaluate_crop_yield_for_hao(self.weather, self.crop_config, grid,
                radiation, scenario_id=eval_key, provenance={"source":"Radiance + representative sampling", "cache":str(folder)},
                spatial_mode=spatial_mode, baseline=natural)
            fresh = lambda x: None if x is None else x/self.config.grain_dry_matter_fraction/1000.
            array = resolve_array(self.array, design)
            row = economic_metrics(kind, hourly, prices, design, self.module, array, grid,
                fresh(payload["yield_kg_DM_ha"]), fresh(natural["yield_kg_DM_ha"]),
                payload["all_points_mature"], pv_reference, self.config)
            row.update(design_id=fingerprint(asdict(design))[:16], **asdict(design), n_rows=array.n_rows,
                       n_modules=array.n_rows*array.n_mods, field_area_ha=grid.area_m2/10000., sampling_mode=sampling.mode,
                       crop_spatial_mode=spatial_mode, pv_reference_kwh_ha=pv_reference,
                       crop_status=payload["status"], field_validated=False)
            rows.append(row)
        bridge.save_json(evaluation_file, dict(rows=rows, key=eval_key))
        return rows

    def search(self, pop_size, n_gen, seed, bounds, base_design):
        """Run one NSGA-II search per construction, then merge their Pareto sets.

        Radiance results are shared and cached, so the two optimizations do not
        repeat the expensive physics calculation for a design they both visit.
        """
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize

        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 2:
            raise ValueError("NSGA-II pop_size must be an integer >= 2")
        if isinstance(n_gen, bool) or not isinstance(n_gen, int) or n_gen < 1:
            raise ValueError("NSGA-II n_gen must be a positive integer")
        names = list(bounds)
        lo = np.asarray([bounds[k][0] for k in names], float)
        hi = np.asarray([bounds[k][1] for k in names], float)
        if not np.isfinite([lo, hi]).all() or (hi <= lo).any():
            raise ValueError("Invalid NSGA-II decision bounds")
        if set(names) - set(asdict(base_design)):
            raise ValueError("Decision bounds contain unknown design fields")
        if mounting_of(base_design) != self.mounting:
            raise ValueError(f"BASE_DESIGN does not match MOUNTING = {self.mounting!r}")
        depth = float(getattr(self.array, "field_depth_m", 0.) or 0.)
        if depth > 0 and "pitch" in names and bounds["pitch"][1] > depth:
            raise ValueError(f"Upper pitch bound {bounds['pitch'][1]} m exceeds the field depth {depth} m: "
                             "such a design would have no row. Lower the bound or deepen the field.")

        records = {}
        for kind in self.constructions:
            engine = self
            class AgriPVProblem(Problem):
                # Vectorised: pymoo hands over a whole generation at once, so the
                # expensive physics of all its designs can run in parallel processes.
                def __init__(self):
                    super().__init__(n_var=len(names), n_obj=2, n_ieq_constr=1, xl=lo, xu=hi)
                def _evaluate(self, X, out, *args, **kwargs):
                    designs = [replace(base_design, **dict(zip(names, map(float, x)))) for x in np.atleast_2d(X)]
                    engine.precompute(designs)
                    F, G = [], []
                    for design in designs:
                        print(f"NSGA-II {kind} design: {asdict(design)}", flush=True)
                        row = next(r for r in engine.evaluate(design) if r["pv_type"] == kind)
                        records[(row["design_id"], kind)] = row
                        F.append([-float(row["LER"]), -float(row["annual_net_eur_ha"])])
                        G.append([float(row["constraint_violation"])])
                    out["F"], out["G"] = np.asarray(F, float), np.asarray(G, float)

            algorithm = NSGA2(pop_size=pop_size, sampling=FloatRandomSampling(),
                crossover=SBX(eta=15, prob=.9), mutation=PM(eta=20), eliminate_duplicates=True)
            print(f"Starting NSGA-II: {kind}, pop_size={pop_size}, n_gen={n_gen}")
            minimize(AgriPVProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=True)

        table = mark_pareto(pd.DataFrame(records.values()))
        table["optimization_method"] = "NSGA-II; separate construction runs; representative irradiance sampling"
        return table.sort_values(["pareto", "pv_type", "LER"], ascending=[False, True, False]).reset_index(drop=True)

    def export(self, table):
        folder = self._results_folder(f"nsga2_{self.sampling.mode}")
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder/"evaluated_candidates.csv", index=False)
        table[table.pareto & table.feasible].to_csv(folder/"pareto_front.csv", index=False)
        bridge.save_json(folder/"run_metadata.json", dict(
            mounting=self.mounting, sampling=asdict(self.sampling), config=asdict(self.config), price_assumptions=self.price_assumptions,
            weather=self.weather_metadata, reference=asdict(self.reference_design),
            method="NSGA-II, run separately for each construction; representative irradiance weighted by target/sample GHI",
            limitations=["Best evaluated design, not global optimum", "Sampling requires full-hour verification",
                         "Shortwave coupling; crop diffuse reconstructed internally; no shade microclimate",
                         "PP crop uncalibrated; effective planted fraction is not a spatial support/path mask",
                         "Economics and module data assumed; constant temperature derate; annual net deducts annuitised CAPEX",
                         "Constant repeated annual cashflow, no financing/tax/replacements/degradation; payback uses operating cashflow",
                         "Small finite array has edge effects; plot rotates with array, not a fixed parcel"] ))
        if not table.feasible.any(): print("No feasible evaluated candidates; inspect constraints/calendar, do not select an infeasible row")
        print("Saved NSGA-II Pareto results:", folder)

    def _results_folder(self, name):
        # Fixed tilt keeps the v6 folder names; single-axis/vertical results never overwrite them.
        prefix = "" if self.mounting == "fixed" else f"{self.mounting}_"
        # Winter wheat (the reference crop) keeps the existing names; other crops get their own folders.
        c = self.crop_config
        crop = "" if (c.crop_name, c.variety) == ("wheat", "Winter_wheat_103") else f"_{c.crop_name}_{c.variety}"
        return self.root/"results"/f"{prefix}{name}{crop}"

    def _design_fields(self):
        cls = self.ns[DESIGN_CLASSES[self.mounting]]
        return cls, [f.name for f in fields(cls) if f.init]

    def _design_from_row(self, row):
        cls, _ = self._design_fields()
        return build_design(cls, row)

    def n_skies(self, sampling):
        """Number of distinct sampled skies per construction and design."""
        plan = make_sampling_plan(self.hours, self.hourly_weather.GHI, self.hourly_weather.DHI, sampling)
        return int(plan.sample_hour.nunique())

    def recheck(self, source, sampling, spatial_mode="area_mean_approximation"):
        """Stage 2: re-evaluate the designs of a search result with a denser sampling.

        `source` is the Pareto table (or any table with the seven design columns and
        design_id). Every distinct design is evaluated for BOTH constructions, so the
        new Pareto sets are formed among these designs only. Radiance runs in parallel
        worker processes exactly as in the search, and finished designs stay cached.
        Writes results/recheck_<mode>_<n>skies/ and returns the new table.
        """
        source = pd.DataFrame(source)
        _, names = self._design_fields()
        missing = [c for c in ["design_id", "pv_type", *names] if c not in source.columns]
        if missing:
            raise ValueError(f"Source table lacks columns: {missing}")
        unique = source.drop_duplicates("design_id")
        designs = [self._design_from_row(row) for _, row in unique.iterrows()]
        if not designs:
            raise ValueError("No designs to re-check")
        rebuilt = [fingerprint(asdict(d))[:16] for d in designs]
        lost = int((np.asarray(rebuilt) != unique.design_id.astype(str).to_numpy()).sum())
        if lost:
            raise ValueError(f"{lost} design(s) do not reproduce their design_id: read the CSV with "
                             "pd.read_csv(..., float_precision='round_trip') so the cache and the comparison match")
        self.precompute(designs, sampling)
        rows = []
        for i, design in enumerate(designs, 1):
            print(f"Re-check {i}/{len(designs)}: {asdict(design)}", flush=True)
            rows.extend(self.evaluate(design, sampling, spatial_mode))
        skies = self.n_skies(sampling)
        table = mark_pareto(pd.DataFrame(rows))
        table["n_skies"] = skies
        table["optimization_method"] = "re-check of search designs with denser representative sampling"
        table = table.sort_values(["pareto", "pv_type", "LER"], ascending=[False, True, False]).reset_index(drop=True)

        keys = ["design_id", "pv_type"]
        metrics = ["LER", "LER_crop", "LER_pv", "crop_yield_t_ha", "electrical_kwh", "annual_net_eur_ha"]
        have = [m for m in metrics if m in source.columns]
        merged = source[keys + have].drop_duplicates(keys).merge(
            table[keys + have + ["n_rows", "pareto"]], on=keys, suffixes=("_search", "_recheck"))
        for m in have:
            a, b = merged[f"{m}_search"].astype(float), merged[f"{m}_recheck"].astype(float)
            merged[f"{m}_change_%"] = 100.*(b/a - 1.)
        folder = self._results_folder(f"recheck_{sampling.mode}_{skies}skies")
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder/"evaluated_candidates.csv", index=False)
        table[table.pareto & table.feasible].to_csv(folder/"pareto_front.csv", index=False)
        merged.to_csv(folder/"comparison_with_search.csv", index=False)
        bridge.save_json(folder/"run_metadata.json", dict(
            mounting=self.mounting, config=asdict(self.config), sampling=asdict(sampling), n_skies=skies, search_sampling=asdict(self.sampling),
            price_assumptions=self.price_assumptions, reference=asdict(self.reference_design),
            note="Same designs as the search, denser sampling; Pareto sets formed among these designs only."))
        print("Saved re-check results:", folder)
        return table

    def validate_finalists(self, ranked, count):
        if not isinstance(count,int) or count < 1: raise ValueError("Positive validation count required")
        chosen = ranked[ranked.feasible].drop_duplicates("design_id").head(count)
        if chosen.empty: return pd.DataFrame()
        approx_rows, full_mean_rows, full_point_rows = [], [], []
        for _, row in chosen.iterrows():
            design = self._design_from_row(row)
            approx_rows.extend(self.evaluate(design, self.sampling, "area_mean_approximation"))
            full_mean_rows.extend(self.evaluate(design, SamplingConfig(mode="full"), "area_mean_approximation"))
            # Cached full-hour radiation reused: ONLY crop simulations are repeated.
            full_point_rows.extend(self.evaluate(design, SamplingConfig(mode="full"), "pointwise"))
        comparisons = compare_sampling(pd.DataFrame(approx_rows), pd.DataFrame(full_mean_rows))
        comparisons["comparison"] = "sampling_only_same_mean_crop"
        spatial = compare_sampling(pd.DataFrame(full_mean_rows), pd.DataFrame(full_point_rows))
        spatial["comparison"] = "spatial_mean_vs_pointwise_full_hours"
        combined = pd.concat([comparisons,spatial], ignore_index=True)
        folder = self._results_folder(f"nsga2_{self.sampling.mode}")
        folder.mkdir(parents=True, exist_ok=True)
        combined.to_csv(folder/"validation_errors.csv", index=False)
        pd.DataFrame(full_point_rows).to_csv(folder/"full_hour_pointwise_finalists.csv", index=False)
        return combined


ALL_CONSTRUCTIONS = ("semi_transparent", "conventional")


class ConstructionEngine(WeightedEngine):
    """WeightedEngine that ray-traces and optimises only the selected constructions,
    e.g. ("conventional",) to halve (or better) the cost of a single-axis study.

    The base class is left untouched, so runs with both constructions keep their cache
    keys. Here the overriding methods and the selection enter the physics signature,
    so restricted runs get their own cache entries and never mix with full ones."""

    def __init__(self, *args, constructions=ALL_CONSTRUCTIONS, **kwargs):
        chosen = tuple(k for k in ALL_CONSTRUCTIONS if k in set(constructions))
        if not chosen or set(constructions) - set(ALL_CONSTRUCTIONS):
            raise ValueError(f"constructions must be a non-empty subset of {ALL_CONSTRUCTIONS}")
        super().__init__(*args, **kwargs)
        self.constructions = chosen
        extra = dict(constructions=list(chosen), engine_code_subclass={
            "ConstructionEngine._simulation": _code_digest(ConstructionEngine._simulation.__code__),
            "ConstructionEngine._grids": _code_digest(ConstructionEngine._grids.__code__)})
        self.physics_signature.update(extra)
        self.signature.update(extra)
        print("Constructions:", ", ".join(chosen))

    def _grids(self, design):
        return {kind: design_grid(design, self.module, self.array, kind,
                                  self.scenario.ground_grid_x, self.scenario.ground_grid_y,
                                  self.ns["SETTINGS"].ground_sensor_height)
                for kind in self.constructions}

    def _simulation(self, design, sampling, crop=True):
        key = self._physics_key(design, sampling, crop)
        folder = self.cache/key
        manifest_path = folder/"manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if all((folder/p).exists() and bridge.sha256(folder/p) == sha for p, sha in manifest["files"].items()):
                hourly = pd.read_csv(folder/"hourly.csv.gz")
                grounds = {k: pd.read_csv(folder/f"ground_{k}.csv.gz") for k in self._grids(design)} if crop else {}
                print("Physics cache hit:", key[:12])
                return hourly, grounds, folder
            raise RuntimeError(f"Cache checksum mismatch: {folder}; remove this entry and rerun")
        grids = self._grids(design)
        t0 = time.time()
        hourly, grounds = self.ns["model_for_optimization"](
            design, self.scenario, self.module, resolve_array(self.array, design), calculate_linrui=crop, calculate_hao=True,
            ground_grid=grids, sampling_config=sampling, crop_dates=self.pv_days,
            constructions_to_run=self.constructions if crop else ("conventional",), persist_outputs=False,
            ground_components=self.config.trace_ground_components)
        if len(hourly) != 8760:
            raise RuntimeError("Annual hourly result must have 8760 rows")
        folder.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(folder/"hourly.csv.gz", index=False)
        for kind, ground in grounds.items(): ground.to_csv(folder/f"ground_{kind}.csv.gz", index=False)
        files = ["hourly.csv.gz"]+[f"ground_{k}.csv.gz" for k in grounds]
        manifest = dict(key=key, files={p: bridge.sha256(folder/p) for p in files}, elapsed_s=time.time()-t0,
                        sampling=asdict(sampling), design=asdict(design), grids={k: asdict(v) for k, v in grids.items()},
                        constructions=list(self.constructions))
        bridge.save_json(manifest_path, manifest)
        return hourly, grounds, folder


def make_engine(*args, constructions=ALL_CONSTRUCTIONS, **kwargs):
    """The base engine for both constructions (v6 cache keys), else ConstructionEngine."""
    if set(constructions) == set(ALL_CONSTRUCTIONS):
        return WeightedEngine(*args, **kwargs)
    return ConstructionEngine(*args, constructions=constructions, **kwargs)
