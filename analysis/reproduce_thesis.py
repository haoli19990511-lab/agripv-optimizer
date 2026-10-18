"""Reproduce the tables and figures of the thesis from the stored results (no new ray tracing).

    python analysis/reproduce_thesis.py            run every step, then check the key numbers against the thesis
    python analysis/reproduce_thesis.py --list     show which step produces which table or figure
    python analysis/reproduce_thesis.py --only 6   run single steps (numbers as in --list), e.g. --only 6,7
    python analysis/reproduce_thesis.py --check    only compare the key numbers of the existing outputs

Needs the results of the searches and re-checks in results/ (your own runs, or thesis_data.zip unpacked into the
optimizer folder, see README.md). Every step is an ordinary script of analysis/ and can also be run on its own.
Steps 12-13 run WOFOST again from the cached ground irradiation (a few minutes each); all others take seconds.
"""
import json
import subprocess
import sys
import time

import pandas as pd

from common import RESULTS, ROOT

FIXED = "recheck_combined_192skies"
# (script and arguments, thesis items, outputs)
STEPS = [
    (["workflow_diagram.py"], "Figure 3-1", "results/fig_workflow.png"),
    (["results_fixed.py", FIXED], "Figures 4-1, 4-2; monthly capture price for Table 8",
     f"results/{FIXED}/analysis/fig_fixed_pareto_*.png, fixed_pareto_table.csv, monthly_capture_price.csv"),
    (["sensitivity.py", FIXED], "inputs of Table 9; break-even module price (Section 4.6)",
     f"results/{FIXED}/analysis/sensitivity_oat.csv, break_even_module_price.csv, tornado_net_income.png"),
    (["price_scenarios.py", FIXED], "inputs of Table 10 (price years, curtailment)",
     f"results/{FIXED}/analysis/price_scenarios*.csv"),
    (["cost_scenarios.py", FIXED], "cost scenarios and break-even electricity prices (Section 4.6)",
     f"results/{FIXED}/analysis/cost_scenarios*.csv, fig_cost_scenarios.png"),
    (["final_tables.py"], "Tables 4, 5, 6, 8, 9, 10, 15 and the numbers quoted in Chapter 4",
     "results/final_tables/table04_*.csv ... table15_*.csv, numbers.json"),
    (["compare_mountings.py"], "Figure 4-4 (all mounting types)", "results/analysis_mountings/fig_mountings_pareto.png"),
    (["compare_mountings.py", "Fixed tilt,Single-axis"], "Figure 4-3 (fixed tilt and single-axis)",
     "results/analysis_mountings/fig_mountings_pareto_fixed_single-axis.png"),
    (["crop_response.py"], "Figure 4-5 (crop response to shading, Section 4.4)",
     "results/analysis_crop_response/fig_crop_response.png, crop_response_summary.json"),
    (["crop_comparison.py"], "Table 7, Figure 4-6 (comparison of crops, Section 4.5)",
     "results/analysis_crops/crop_summary.csv, fig_crop_comparison.png"),
    (["validation_nrw.py"], "Table 13 (validation against all daylight hours)",
     "results/sampling_validation/errors_vs_full_nrw.csv"),
    (["convergence.py", "single_axis_nsga2_combined", "results/search_single_axis.log"],
     "convergence of the single-axis search (Section 4.7)", "results/single_axis_nsga2_combined/analysis/hypervolume_*"),
    (["water_limited_check.py"], "Table 14 (water-limited crop model, Section 4.8)",
     "results/water_limited_check/water_limited_check.csv"),
]

# Key numbers of the thesis: (description, function returning the value, value in the thesis, tolerance)
def _numbers():
    return json.loads((RESULTS/"final_tables"/"numbers.json").read_text())


def _validation(n_skies):
    e = pd.read_csv(RESULTS/"sampling_validation"/"errors_vs_full_nrw.csv")
    return e[e.n_skies == n_skies].net_diff_eur_ha


def _water(crop, col):
    w = pd.read_csv(RESULTS/"water_limited_check"/"water_limited_check.csv")
    return w[(w.crop == crop) & (w.case != "open field")][col]


CHECKS = [
    ("Fixed tilt: conventional Pareto designs (Table 4)", lambda: _numbers()["fixed"]["n_pareto_conv"], 17, 0),
    ("Fixed tilt: highest net income, EUR/(ha a)", lambda: _numbers()["fixed"]["best"], 954, .5),
    ("Fixed tilt: net income at the highest LER", lambda: _numbers()["fixed"]["top_net"], -27832, .5),
    ("Fixed tilt: lowest / highest LER", lambda: [round(v, 2) for v in _numbers()["fixed"]["conv_LER"]], [1.21, 1.57], 0),
    ("Fixed tilt: best without agri-PV bonus", lambda: _numbers()["fixed"]["best_no_bonus"], -1808, .5),
    ("Single-axis: highest net income", lambda: _numbers()["single"]["best"], -2964, .5),
    ("Vertical: highest net income", lambda: _numbers()["vertical"]["best"], 2578, .5),
    ("Vertical: break-even structure cost, EUR/kWp", lambda: _numbers()["vertical"]["best_break_even"], 132, .5),
    ("Break-even electricity price of the best design, ct/kWh", lambda: _numbers()["break_even_price_ct_best"], 6.0, .05),
    ("Sensitivity: minimum Jaccard index of the Pareto set", lambda: _numbers()["sensitivity_jaccard_min"], .78, .005),
    ("Table 13: net-income bias with 192 skies, min", lambda: _validation(192).min(), 210, 5),
    ("Table 13: net-income bias with 192 skies, max", lambda: _validation(192).max(), 440, 5),
    ("Table 14: wheat, shallow soil, relative yield min", lambda: _water("winter_wheat", "rel_WLP_shallow").min(), 1.10, .005),
    ("Table 14: wheat, shallow soil, relative yield max", lambda: _water("winter_wheat", "rel_WLP_shallow").max(), 1.14, .005),
]


def check():
    failed = 0
    print(f"\n{'Key number':58s} {'reproduced':>14s} {'thesis':>12s}  result")
    for text, get, expected, tol in CHECKS:
        try:
            value = get()
            if isinstance(expected, list):
                ok = value == expected
            else:
                value = float(value)
                ok = abs(value-expected) <= tol
        except FileNotFoundError as e:
            value, ok = f"missing ({e.filename.split('results')[-1]})", False
        failed += not ok
        shown = value if isinstance(value, (list, str)) else round(value, 3)
        print(f"{text:58s} {str(shown):>14s} {str(expected):>12s}  {'OK' if ok else 'DIFFERENT'}")
    print(f"\n{len(CHECKS)-failed} of {len(CHECKS)} key numbers agree with the thesis.")
    return failed


def main(argv):
    if "--list" in argv:
        for i, (cmd, items, outputs) in enumerate(STEPS, 1):
            print(f"{i:2d}. python analysis/{' '.join(cmd)}\n      {items}\n      -> {outputs}")
        return 0
    if "--check" in argv:
        return 1 if check() else 0
    only = None
    if "--only" in argv:
        only = {int(x) for x in argv[argv.index("--only")+1].split(",")}
    for i, (cmd, items, _) in enumerate(STEPS, 1):
        if only and i not in only:
            continue
        print(f"\n=== Step {i}: {items}\n    python analysis/{' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT/"analysis"/cmd[0]), *cmd[1:]], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:])
            print(f"Step {i} failed (exit {r.returncode}).")
            return r.returncode
        print(f"    done in {time.time()-t0:.0f} s")
    return 1 if (only is None and check()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
