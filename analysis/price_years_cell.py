# Paste into a NEW cell at the END of agripv_optimizer_fixed.ipynb, AFTER the search has finished.
# Re-evaluates the Pareto designs with other electricity-price years. Radiance results come
# from the cache (same SAMPLING), so only WOFOST + economics run: minutes, not hours.
# Output: results/nsga2_<mode>/analysis/price_year_scenarios.csv
from pathlib import Path
import pandas as pd

PRICE_SCENARIOS = {
    "2023-2025 (base)": [2023, 2024, 2025],
    "2020-2022 (crisis)": [2020, 2021, 2022],
    "2020-2025 (all)": [2020, 2021, 2022, 2023, 2024, 2025],
    "2025 only": [2025],
}
run_folder = Path("results")/f"nsga2_{SAMPLING.mode}"
designs = pd.read_csv(run_folder/"pareto_front.csv", float_precision="round_trip").drop_duplicates("design_id")
design_fields = list(BASE_DESIGN.__dataclass_fields__)

rows = []
for name, years in PRICE_SCENARIOS.items():
    e = WeightedEngine(globals(), CROP_CONFIG, ENGINE_CONFIG, SAMPLING, REFERENCE_PV_DESIGN,
                       "csv", FLAT_PRICE_EUR_KWH, years, parallel_designs=1, notebook_path=NOTEBOOK_FILE)
    for _, d in designs.iterrows():
        design = PVDesign(**{k: float(d[k]) for k in design_fields})  # fixed tilt
        for r in e.evaluate(design):
            rows.append(dict(price_scenario=name, design_id=r["design_id"], pv_type=r["pv_type"],
                             LER=r["LER"], electrical_revenue_eur=r["electrical_revenue_eur"],
                             annual_net_eur_ha=r["annual_net_eur_ha"],
                             payback_period_years=r["payback_period_years"]))
price_table = pd.DataFrame(rows)
(run_folder/"analysis").mkdir(exist_ok=True)
price_table.to_csv(run_folder/"analysis"/"price_year_scenarios.csv", index=False)
display(price_table.pivot_table(index=["pv_type", "design_id"], columns="price_scenario",
                                values="annual_net_eur_ha").round(0))
