"""Shared helpers for post-processing NSGA-II results (no Radiance / WOFOST needed).

Economics are recomputed from the per-design columns in evaluated_candidates.csv with
exactly the formula of weighted_engine.economic_metrics, so cost/price assumptions can be
varied in seconds. Changing the PRICE YEARS still needs hourly kWh (see price_years_cell.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/"results"

# Reference palette slots 1/2 (blue, orange); grey for context marks.
COLORS = {"conventional": "#2a78d6", "semi_transparent": "#eb6834"}
LABELS = {"conventional": "Conventional", "semi_transparent": "Semi-transparent"}
GREY = "#b5b4ae"
DECISIONS = ["tilt", "pitch", "azimuth", "clearance_height"]
TRACKING_DECISIONS = ["pitch", "axis_azimuth", "hub_height", "limit_angle"]


VERTICAL_DECISIONS = ["pitch", "azimuth", "clearance_height"]


def decisions(table):
    """Decision variables of a results table (fixed tilt, single-axis tracking or vertical)."""
    if "axis_azimuth" in table.columns:
        return TRACKING_DECISIONS
    if "mounting" in table.columns and (table.mounting == "vertical").all():
        return VERTICAL_DECISIONS
    return DECISIONS


def geometry_columns(t):
    """Mounting, structure height and lowest edge per row; rows from v6 tables (or pooled with
    them) lack the v7 columns and are fixed-tilt designs whose clearance is all three heights."""
    mounting = t.mounting.fillna("fixed") if "mounting" in t.columns else pd.Series("fixed", index=t.index)
    if "clearance_height" not in t.columns:  # single-axis tables always carry the v7 columns
        return mounting, t.structure_height_m, t.min_clearance_m
    height = t.structure_height_m.fillna(t.clearance_height) if "structure_height_m" in t.columns else t.clearance_height
    low = t.min_clearance_m.fillna(t.clearance_height) if "min_clearance_m" in t.columns else t.clearance_height
    return mounting, height, low


def structure_rate(t, c):
    """Structure EUR/kWp per row, as in weighted_engine.economic_metrics (v6 and v7)."""
    mounting, height, _ = geometry_columns(t)
    rate = c["structure_base_eur_kwp"] + c["structure_extra_eur_kwp_m"]*np.maximum(
        0., height-c["structure_reference_height_m"])
    rate = rate + np.where(mounting == "single_axis", c.get("tracker_extra_eur_kwp", 150.), 0.)
    return np.where(mounting == "vertical", c.get("vertical_structure_eur_kwp", 50.), rate)
KPIS = ["LER", "LER_crop", "LER_pv", "annual_net_eur_ha"]

ECON_KEYS = ["module_semi_eur_kwp", "module_conv_eur_kwp", "structure_base_eur_kwp",
             "structure_reference_height_m", "structure_extra_eur_kwp_m", "opex_fraction",
             "crop_price_eur_t", "crop_cost_eur_planted_ha", "lifetime_years", "discount_rate",
             "planted_fraction"]


def load_run(folder):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = RESULTS/folder
    table = pd.read_csv(folder/"evaluated_candidates.csv", float_precision="round_trip")
    meta = json.loads((folder/"run_metadata.json").read_text())
    return table, meta, folder


def crf(rate, years):
    return 1/years if rate == 0 else rate/(1-(1+rate)**(-years))


def recompute_economics(table, config, electricity_price_factor=1.0):
    """Return a copy with annual_net_eur_ha etc. recomputed under `config` (dict).

    electricity_price_factor scales the whole hourly price profile uniformly."""
    t = table.copy()
    c = config
    structure = structure_rate(t, c)
    module_rate = np.where(t.pv_type == "semi_transparent", c["module_semi_eur_kwp"], c["module_conv_eur_kwp"])
    # other_capex_eur_kwp (inverters, electrical BOS, installation, fees, site preparation) is not
    # in the engine; 0 when absent, so the engine's numbers are reproduced exactly.
    capex = t.installed_kwp*(module_rate+structure+c.get("other_capex_eur_kwp", 0.))
    opex = capex*c["opex_fraction"]
    # Planted share as evaluated by the engine (uniform for elevated racks, row strips for vertical).
    planted = t.planted_area_ha if "planted_area_ha" in t.columns else t.footprint_ha*c["planted_fraction"]
    elec = t.electrical_revenue_eur*electricity_price_factor
    crop_rev = t.crop_yield_t_ha.fillna(0.)*planted*c["crop_price_eur_t"]
    crop_cost = planted*c["crop_cost_eur_planted_ha"]
    # Agri-PV bonus as in the engine: rate x kWh for non-vertical designs with >= 2.1 m clearance.
    # Configs without the key (runs before the bonus existed) give 0, so old results reproduce.
    mounting, _, clearance = geometry_columns(t)
    eligible = (clearance >= c.get("agripv_bonus_min_clearance_m", 2.1)) & (mounting != "vertical")
    bonus = c.get("agripv_bonus_eur_kwh", 0.)*t.electrical_kwh*eligible
    cashflow = elec+bonus+crop_rev-opex-crop_cost
    annual_net = cashflow-capex*crf(c["discount_rate"], c["lifetime_years"])
    t["electrical_revenue_eur"] = elec
    t["capex_total_eur"] = capex
    t["annual_operating_cashflow_eur"] = cashflow
    t["annual_net_eur"] = annual_net
    t["annual_net_eur_ha"] = annual_net/t.footprint_ha
    t["payback_period_years"] = np.where(cashflow > 0, capex/cashflow, np.nan)
    return t


def check_reproduction(table, config):
    """The recomputation must reproduce the engine's numbers exactly."""
    t = recompute_economics(table, config)
    err = np.max(np.abs(t.annual_net_eur_ha-table.annual_net_eur_ha))
    if err > 1e-6*max(1., table.annual_net_eur_ha.abs().max()):
        raise AssertionError(f"Economics recomputation differs from engine by {err:.3g} EUR/ha")
    return err


def pareto_mask(values):
    """Non-dominated rows for maximisation of every column."""
    v = np.asarray(values, float)
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        dom = np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1)
        keep[i] = not dom.any()
    return keep


def mark_pareto(table, objectives=("LER", "annual_net_eur_ha")):
    t = table.copy()
    t["pareto"] = False
    for _, g in t.groupby("pv_type"):
        g = g[g.feasible.astype(bool)]
        if len(g):
            t.loc[g.index[pareto_mask(g[list(objectives)])], "pareto"] = True
    return t


def hypervolume_2d(points, ref):
    """Hypervolume of maximisation points (n x 2) above reference point `ref`."""
    p = np.asarray(points, float)
    p = p[(p[:, 0] > ref[0]) & (p[:, 1] > ref[1])]
    if not len(p):
        return 0.
    p = p[pareto_mask(p)]
    p = p[np.argsort(-p[:, 0])]  # descending objective 1, so objective 2 ascends
    hv, prev_y = 0., ref[1]
    for x, y in p:
        hv += (x-ref[0])*(y-prev_y)
        prev_y = y
    return hv
