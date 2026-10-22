# Agri-PV design optimiser: NSGA-II with Radiance and WOFOST

This repository contains the optimisation framework of the bachelor thesis

> Hao Li (2026): *Multi-Objective Optimisation of Agrivoltaic System Design: Balancing Land Equivalent Ratio and
> Revenue.* Bachelor thesis, FH Aachen University of Applied Sciences, Campus Jülich, Solar-Institut Jülich (SIJ).
> Supervisor: Prof. Dr.-Ing. Spiros Alexopoulos.

For each candidate design (row pitch, tilt, azimuth, clearance, ...), the framework ray-traces the irradiance on
the PV modules and on the ground, simulates the crop yield under the array, computes the annual electricity
yield, revenue and costs, and lets the genetic algorithm NSGA-II search for the Pareto front of two objectives:
the **land equivalent ratio (LER)** and the **annual net income per hectare**. Fixed-tilt racks, horizontal
single-axis trackers and vertical bifacial fences are supported.

The framework couples the models of a collaborative project of three theses:

| Part | Author | Files |
|---|---|---|
| Radiation model (bifacial_radiance / Radiance), three mounting types | Tianze Liu | Section A of the notebooks |
| Crop model set-up (WOFOST 7.2 in PCSE, parameters, verification) | Linrui Li | `bridge.py` (model set-up), `inputs/precipitation/` (rainfall data) |
| Optimisation layer: sampling, coupling and interface, economics, constraints, caching, parallel runs, NSGA-II, re-check, crop selection, all analyses | Hao Li | everything else |

---

## 1. Installation

Tested on Windows 11 with Python 3.10.20, Radiance 6.1, bifacial_radiance 0.5.3, pvlib 0.15.2, pymoo 0.6.2,
PCSE 6.0.13, numpy 2.2.6 and pandas 2.3.3.

1. **Radiance.** Install Radiance 6.x (https://github.com/LBNL-ETA/Radiance/releases). The notebooks look for it in
   `%USERPROFILE%\Radiance`, `/usr/local/radiance` or `/opt/homebrew/opt/radiance` and add its `bin` and `lib`
   folders to `PATH` and `RAYPATH` (Cell 1 of Section A).
2. **Python environment.**
   ```
   conda create -n agripv python=3.10
   conda activate agripv
   pip install -r requirements.txt
   ```
3. **Check the installation** (from the repository folder, about one minute):
   ```
   python test_weighted.py
   python test_crops.py
   python test_tracking.py
   python test_parallel.py
   ```

Ray tracing is only needed for new designs. To reproduce the thesis from stored results, Radiance is not
required (Section 6).

---

## 2. Repository structure

```
agripv_optimizer_fixed.ipynb          notebook for fixed-tilt racks            (results in results/)
agripv_optimizer_single_axis.ipynb    notebook for single-axis trackers        (results in results/single_axis_*)
agripv_optimizer_vertical.ipynb       notebook for vertical bifacial fences    (results in results/vertical_*)
thesis_figures.ipynb                  regenerates the figures of the thesis from the stored results
weighted_engine.py    engine: economics and constraints (economic_metrics), caches, parallel runs, NSGA-II, re-check
sampling_core.py      representative sky sampling and transfer to all hours (Eq. 3.9 of the thesis)
bridge.py             crop-model interface: daily weather, crop library, WOFOST/PCSE run
crops.py              crop presets (calendar, dry-matter fraction, price and production cost)
eval_worker.py        worker process for the parallel ray tracing of one design
analysis/             headless runs, all tables and figures of the thesis, reproduction scripts
inputs/crops/         official WOFOST 7.2 crop parameter files (see PROVENANCE.json)
inputs/precipitation/ ERA5 rainfall for the months of the weather file (water-limited check)
weather/              PVGIS typical meteorological year for Aachen (EPW)
prices/               hourly day-ahead prices of the DE-LU bidding zone 2020-2025 (ENTSO-E)
test_*.py             51 unit and integration tests
results/              outputs and caches (not in git; created by the runs, or unpacked from thesis_data.zip)
```

### The notebooks

The three notebooks have the same structure; only the mounting type (Section C) differs.

| Section | Content | Edit? |
|---|---|---|
| A | Radiation model of Tianze Liu, adapted to sampled skies (scene, sky, sensors, glass, tracker, vertical) | no |
| A2 | Study field (`FIELD_PRESET`), module, ground grid, ray-tracing scope | field size only |
| B | Crop-model interface and crop selection | no |
| C | Optimisation problem: economic settings, mounting type, module constructions, decision variables and bounds | bounds, constructions |
| D | **Control panel**: search settings, crop, sampling, parallel runs, prices, reference plant; starts the search | yes |
| E | Pareto front of the search | – |
| F | **Re-check** of Pareto designs with a denser sampling (the values reported in the thesis) | yes |

---

## 3. Running the optimiser

### In the notebook

1. Open the notebook of the mounting type and run all cells with `RUN_SEARCH = False` (setup only).
2. Edit Section D (crop, sampling, population, generations, seed) and **save the notebook (Ctrl+S)**: the parallel
   workers re-read the file and refuse to run if the model code differs from the running kernel.
3. Set `RUN_SEARCH = True` and run Section D. Results go to `results/nsga2_<mode>/` (fixed tilt),
   `results/single_axis_nsga2_<mode>/` or `results/vertical_nsga2_<mode>/`.
4. Re-check the Pareto designs in Section F, e.g.
   ```python
   RUN_RECHECK = True
   RECHECK_SAMPLING = SamplingConfig(mode="combined", hours_per_day=8, days_per_month=2)   # 192 skies
   RECHECK_SOURCE = r"results\fixed_pooled_bonus_pareto.csv"
   ```
   Results go to `results/recheck_combined_192skies/` (other crops: `..._<crop>_<variety>`).
   Set `RECHECK_SOURCE` explicitly: with `None`, the newest `nsga2_*` file is used, which may be a different run.

### Without Jupyter

```
python -u analysis/run_search.py agripv_optimizer_vertical.ipynb > results/search_vertical.log
python -u analysis/run_search.py agripv_optimizer_fixed.ipynb --seed 2 --tag seed2_conv_bonus --conventional-only
python -u analysis/run_recheck.py results/fixed_pooled_bonus_pareto.csv 2 8 agripv_optimizer_fixed.ipynb
python -u analysis/run_recheck.py results/vertical_nsga2_combined/pareto_front.csv 2 8 agripv_optimizer_vertical.ipynb --crop potato
python -u analysis/validate_sampling.py
```
The scripts execute the model cells and the control panel of the given notebook, so the notebook settings apply.
`validate_sampling.py` evaluates the three fixed-tilt designs of Table 13 with all 4,306 daylight hours (18–36 h
per design); its full-year results are included in `thesis_data.zip`.

### Computing time (Intel i5-13600KF, 20 threads, five designs in parallel)

| Task | Time |
|---|---|
| One fixed-tilt design, 24 skies (search) | about 12 min |
| One design, 192 skies (re-check) | about 3 h |
| One design, all 4,306 daylight hours | 18–36 h |
| Search, 20 designs × 6 generations | about 3 h (fixed tilt, conventional modules, 24 skies) to 12 h (vertical, 96 skies) |
| Change of crop, prices or costs for stored designs | seconds to minutes |

---

## 4. What can be changed where

The physics cache stores the ray tracing of every design under a key that contains only what shapes the Radiance
run (geometry, weather file, sampling, module, field and the byte code of the model functions). Everything else
can be changed afterwards **without new ray tracing**.

### No new ray tracing (seconds to minutes)

| Change | Where |
|---|---|
| Crop (8 presets, any crop of the WOFOST 7.2 library) | Section D: `CROP = "potato"`; `CROP_OVERRIDES = dict(variety=..., start=...)` |
| Crop price and production cost | `crops.py`, `CROP_PRESETS` |
| Investment, O&M, discount rate, lifetime, agri-PV bonus | `weighted_engine.py`, `EngineConfig`; or in the notebook: `ENGINE_CONFIG = replace(ENGINE_CONFIG, discount_rate=.07)` |
| Constraints (crop retention 66 %, clearance 2.5 m, corridor 1.5 m) | `EngineConfig`: `minimum_crop_retention`, `minimum_clearance_m`, `minimum_clear_corridor_m` |
| Electricity price years or a flat price | Section D: `PRICE_YEARS`, `PRICE_MODE = "flat_assumption"`, `FLAT_PRICE_EUR_KWH` |
| Designs to re-check | Section F: `RECHECK_SOURCE` |

### New ray tracing (hours per design)

| Change | Where |
|---|---|
| Sampling (days per month × hours per day) | Section D: `SAMPLING`; Section F: `RECHECK_SAMPLING` |
| Field size and modules per row | Section A2: `FIELD_PRESET` (`"small"`, `"large"`, `"custom"` with `CUSTOM_N_MODS`, `CUSTOM_FIELD_DEPTH_M`) |
| Ground-grid spacing | Section A2: `GROUND_GRID_SPACING_M` (custom field) |
| Trace every module instead of the middle one | Section A2: `PV_TRACE_MODE = "all"` (very slow) |
| Module constructions | Section C: `CONSTRUCTIONS = ("semi_transparent", "conventional")` |
| Module data, albedo, weather file | Section A2: `module_specification`, `scenario` (use a new file name for new weather data) |

### New search (hours)

| Change | Where |
|---|---|
| Start a search | Section D: `RUN_SEARCH = True` |
| Population, generations, seed | Section D: `POP_SIZE`, `N_GENERATIONS`, `SEED` |
| Decision variables and bounds | Section C: `DECISION_BOUNDS`, `BASE_DESIGN` |
| Number of parallel designs | Section D: `PARALLEL_DESIGNS` |

### Do not change

* The functions of Section A and `sampling_core.py`: they define the physics; any change (even of a docstring in
  `sampling_core.py`) changes the cache key and requires new ray tracing.
* `MOUNTING` in Section C: open the notebook of the other mounting type instead.
* Any notebook or `.py` file while a run is in progress (the workers compare the code and stop).

---

## 5. Input data and sources

| Data | File | Source |
|---|---|---|
| Weather (typical meteorological year, hourly, UTC) | `weather/Aachen_PVGIS_TMY.epw` | PVGIS 5.3, European Commission JRC, years 2005–2023 |
| Day-ahead prices DE-LU 2020–2025, hourly | `prices/electricity_price_<year>_DE_LU.csv` | ENTSO-E Transparency Platform |
| Crop parameters (23 crops) | `inputs/crops/*.yaml` | WOFOST crop parameters, branch wofost72 (De Wit; see `PROVENANCE.json`) |
| Crop prices and production costs | `crops.py` | Landwirtschaftskammer Nordrhein-Westfalen (2024), Planungsdaten 2024/25; soybean, faba bean and oilseed rape are assumptions |
| Rainfall for the months of the weather file | `inputs/precipitation/Aachen_TMY_precipitation_supplement.json` | ERA5 reanalysis via Open-Meteo (CC BY 4.0), prepared by Linrui Li |
| Module | `module_specification` (Section A2) | LONGi LR5-72HGD datasheet (590 Wp, bifaciality 80 ± 5 %) |
| Investment costs | `EngineConfig` | SYMBIOSYST D7.3 (2026); Bellone et al. (2024); see Table 2 of the thesis |

---

## 6. Reproducing the thesis

Every table and figure of Chapter 4 can be regenerated from stored results in a few minutes, **without
ray tracing**. The results are distributed as `thesis_data.zip` (180 MB), attached to the release
[v1.0-thesis](https://github.com/haoli19990511-lab/agripv-optimizer/releases/tag/v1.0-thesis) and to the digital
appendix of the thesis. It contains the result folders of the searches and re-checks and the physics
cache of the 192-sky re-checks and of the full-year validation.

```
# in the repository folder
unzip thesis_data.zip            # creates results/...
python analysis/reproduce_thesis.py
```

`reproduce_thesis.py` runs the analysis scripts in order and finally compares 14 key numbers with the thesis
(e.g. highest net income of fixed tilt 954 EUR/(ha·a), 17 Pareto designs, vertical break-even 132 EUR/kWp, Table 13
and Table 14 ranges). `--list` shows the steps, `--only 6` runs single steps, `--check` only compares.
The package itself is built with `python analysis/export_thesis_data.py`.

In Jupyter, `thesis_figures.ipynb` runs the figure scripts one by one and shows the figures of Chapters 3
and 4, two of the tables and the check of the key numbers (kernel: the `agripv` environment).

| Thesis | Script | Output |
|---|---|---|
| Figure 3-1 | `workflow_diagram.py` | `results/fig_workflow.png` |
| Tables 1–3 | inputs: `DECISION_BOUNDS` (Section C), `EngineConfig`, search settings (Section D) | – |
| Figures 4-1, 4-2 | `results_fixed.py recheck_combined_192skies` | `results/recheck_combined_192skies/analysis/fig_fixed_pareto_*.png` |
| Tables 4, 5, 6, 8, 9, 10, 15 and quoted numbers | `sensitivity.py`, `price_scenarios.py`, `cost_scenarios.py` (on `recheck_combined_192skies`), then `final_tables.py` | `results/final_tables/table04_fixed.csv` … `table15_single_axis.csv`, `numbers.json` |
| Figures 4-3, 4-4 | `compare_mountings.py` | `results/analysis_mountings/` |
| Joint Pareto front of all mounting types (Section 4.3) | `joint_front.py` (after `compare_mountings.py`) | `results/analysis_mountings/joint_front.csv`, `fig_joint_front.png` |
| Figure 4-5 | `crop_response.py` | `results/analysis_crop_response/` |
| Table 7, Figure 4-6 | `crop_comparison.py` | `results/analysis_crops/` |
| Table 13 | `validate_sampling.py` (full-year runs), then `validation_nrw.py` | `results/sampling_validation/` |
| Table 14 | `water_limited_check.py` | `results/water_limited_check/` |
| Convergence and seeds (Section 4.7) | `convergence.py <run> <search log>` | `results/<run>/analysis/hypervolume_*` |
| Tables 11, 12, single-module tracing (Section 4.7) | `accuracy_tests.py grid`, `hours`, `trace` (new ray tracing) | `results/accuracy_tests/` |

Notes:

* The full results of a new run are produced by the commands of Section 3 (searches, re-checks, crops) and
  `validate_sampling.py`; the shell scripts used for the thesis runs are documented in the run logs.
* Figure 4-5 uses the stored WOFOST runs in `runs/aachen_research/` (about 28 GB, not distributed). Without them,
  `crop_response.py` rebuilds the figure from the extracted table `results/analysis_crop_response/crop_response_runs.csv`.
* Tables 11 and 12 and the single-module tracing test of Section 4.7 change settings that enter the ray tracing
  (ground grid of 12/48/96 points; 1/2/4/8 sampled hours per day; `PV_TRACE_MODE = "all"` versus `"center"`).
  They are reproduced with new ray tracing by `analysis/accuracy_tests.py grid|hours|trace` (about 10 min, 1–2 h
  and 5–8 h); `--dry-run` shows the configurations. They are not part of `reproduce_thesis.py`.

---

## 7. Model summary

* **Sampling.** In each month, K representative days (quantiles of the daily irradiation) and on each of them H
  hours (the brightest hour of H blocks of daylight) are ray-traced; every daylight hour of the year is scaled
  from its nearest sampled hour by GHI(t)/GHI(sample). Thesis: 24 skies (K=1, H=2) in the fixed-tilt search,
  96 skies (K=1, H=8) for single-axis and vertical, 192 skies (K=2, H=8) in all re-checks.
* **Electricity.** E = n_modules · A · η · 0.97 · 0.96 · Σ(G_front + 0.8 · G_rear), η = 22.8 %; hourly revenue with
  the mean 2023–2025 price profile; agri-PV bonus 0.5 ct/kWh for clearances ≥ 2.1 m (§ 38b EEG 2023).
* **Crop.** WOFOST 7.2, potential production, driven by the area-weighted daily ground irradiation; LER_crop =
  f_plant · Y_APV / Y_open. A water-limited check (Wofost72_WLP_CWB, ERA5 rainfall) is in `water_limited_check.py`.
* **Objectives.** max LER = LER_crop + LER_PV; max annual net income = (revenue − O&M − CAPEX · CRF) / area,
  5 %, 25 years.
* **Constraints.** crop matures; LER_crop ≥ 0.66 (DIN SPEC 91434); lowest edge ≥ 2.5 m (not vertical);
  corridor ≥ 1.5 m; passed to NSGA-II as one continuous violation.

---

## 8. Limitations

Small study field (3 modules × 24 m) with single-module tracing and no edge effects; crop model not calibrated
for the site and run in potential production; no microclimate; constant temperature derating; no degradation,
financing or taxes; costs partly assumptions. The NSGA-II front is the best evaluated set, not a proven global
optimum. See Section 5.4 of the thesis.

## Licence

The code is released under the MIT licence (see `LICENSE`), © 2026 Hao Li, Tianze Liu, Linrui Li. The data in
`weather/`, `prices/` and `inputs/` remain under the terms of their providers: PVGIS (European Commission, JRC),
ENTSO-E Transparency Platform, WOFOST crop parameters (see `inputs/crops/PROVENANCE.json`) and ERA5 via Open-Meteo
(CC BY 4.0). Please cite the thesis when using the framework.
