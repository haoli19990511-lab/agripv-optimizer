"""Pack the results needed to reproduce every table and figure of the thesis without new ray tracing.

    python analysis/export_thesis_data.py [output.zip]          (default: thesis_data.zip in the optimizer folder)
Contents (paths relative to the optimizer folder, unpack there):
  results/<run>/{evaluated_candidates,pareto_front,comparison_with_search}.csv, run_metadata.json, analysis/
      for the searches, the 192-sky re-checks of the three mounting types and of the five further crops
  results/cache_v5/<key>/  physics cache of the 192-sky re-checks and the full-year validation (hourly module
      irradiance and daily ground irradiation; needed for the price scenarios, crop and water-limited checks)
  results/sampling_validation, analysis_crop_response, fixed_pooled_bonus_pareto.csv, search logs
The runs/ folder (about 28 GB of WOFOST run records) is not included; crop_response.py then uses the extracted
results/analysis_crop_response/crop_response_runs.csv.
"""
import json
import sys
import zipfile
from pathlib import Path

from common import RESULTS, ROOT

CROPS = ["", "_barley_Spring_barley_301", "_maize_Grain_maize_201", "_potato_Potato_701", "_sugarbeet_Sugarbeet_601",
         "_soybean_Soybean_901"]
RECHECKS = [f"{m}recheck_combined_192skies{c}" for m in ("", "single_axis_", "vertical_") for c in CROPS]
SEARCHES = ["nsga2_combined", "single_axis_nsga2_combined", "vertical_nsga2_combined",
            "nsga2_combined_seed1_conv_bonus", "nsga2_combined_seed2_conv_bonus", "nsga2_combined_seed3_conv_bonus"]
RUN_FILES = ["evaluated_candidates.csv", "pareto_front.csv", "comparison_with_search.csv", "run_metadata.json"]
FOLDERS = ["sampling_validation", "analysis_crop_response"]
FILES = ["fixed_pooled_bonus_pareto.csv", "search_single_axis.log", "search_vertical.log",
         "search_seed1.log", "search_seed2.log", "search_seed3.log"]
CACHED_SAMPLINGS = [("combined", 2, 8), ("full", None, None)]  # 192-sky re-check and full-year validation


def wanted_cache_entry(manifest):
    s = manifest.get("sampling", {})
    return any(s.get("mode") == mode and (days is None or (s.get("days_per_month"), s.get("hours_per_day")) == (days, hours))
               for mode, days, hours in CACHED_SAMPLINGS)


def main(out):
    files = []
    for run in RECHECKS + SEARCHES:
        folder = RESULTS/run
        if not folder.exists():
            sys.exit(f"missing result folder: {folder}")
        files += [folder/f for f in RUN_FILES if (folder/f).exists()]
        if (folder/"analysis").exists():
            files += [f for f in (folder/"analysis").rglob("*") if f.is_file() and ".ipynb_checkpoints" not in f.parts]
    for name in FOLDERS:
        files += [f for f in (RESULTS/name).rglob("*") if f.is_file() and ".ipynb_checkpoints" not in f.parts]
    files += [RESULTS/f for f in FILES if (RESULTS/f).exists()]
    n_cache = 0
    for manifest in (RESULTS/"cache_v5").glob("*/manifest.json"):
        if wanted_cache_entry(json.loads(manifest.read_text())):
            files += [f for f in manifest.parent.iterdir() if f.is_file()]
            n_cache += 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    size = Path(out).stat().st_size/1e6
    print(f"{len(files)} files ({n_cache} cache entries), {size:.0f} MB -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"thesis_data.zip")
