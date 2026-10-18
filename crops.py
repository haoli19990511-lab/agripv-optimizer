"""Crop selection: presets for crops of the official WOFOST 7.2 parameter library (inputs/crops).

A preset bundles everything that depends on the crop:
- the WOFOST parameter set (file and variety) and the crop calendar,
- the conversion of the simulated dry storage organ to the harvested fresh product,
- the price and production cost used by the economics.

Calendars are written in the simulated PV year (1 Oct 2001 - 30 Sep 2002). Dates after 30 Sep 2002 are
allowed: the weather is a typical meteorological year, so such days use the weather and ground irradiation
of the same month and day of the PV year (see WeightedEngine.pv_day). A season may last at most one year.

Prices and production costs of winter wheat, spring barley, grain maize, potato and sugar beet are the NRW
planning data (Landwirtschaftskammer Nordrhein-Westfalen 2024, Planungsdeckungsbeitraege 2024/2025): price =
mean producer price 2019-2024, cost = variable costs (direct + variable machinery costs). The other crops keep
ILLUSTRATIVE assumptions that must be replaced by sourced values before their economics are interpreted.
Calendars follow the descriptions of the WOFOST parameter sets (regions and sowing dates in the YAML
metadata); they are not local trials.
"""
from dataclasses import dataclass, replace

import bridge


@dataclass(frozen=True)
class CropPreset:
    label: str
    crop_name: str            # PCSE crop label
    crop_file: str            # file in inputs/crops
    variety: str              # parameter set in that file
    start: str                # sowing or emergence date
    start_type: str           # "sowing" or "emergence"
    end: str                  # latest end (maturity) or harvest date
    end_type: str             # "maturity" or "harvest"
    dry_matter_fraction: float  # dry matter of the harvested fresh product (grain at storage moisture, tubers, beets)
    price_eur_t: float        # price of the fresh harvested product
    cost_eur_ha: float        # production cost per cropped hectare
    note: str = ""


CROP_PRESETS = {
    # Reference crop of the thesis.
    "winter_wheat": CropPreset("Winter wheat", "wheat", "wheat.yaml", "Winter_wheat_103",
                               "2001-10-15", "sowing", "2002-08-31", "maturity", .86, 213.1, 1120.),
    "spring_barley": CropPreset("Spring barley", "barley", "barley.yaml", "Spring_barley_301",
                                "2002-03-26", "sowing", "2002-08-31", "maturity", .86, 196.3, 890.,
                                "EU15 set; sowing late March in the north"),
    "grain_maize": CropPreset("Grain maize", "maize", "maize.yaml", "Grain_maize_201",
                              "2002-05-01", "sowing", "2002-10-31", "maturity", .86, 218.7, 1690.,
                              "calibrated for Luxembourg and Germany: sowing 1 May, maturity about 20 Oct"),
    "potato": CropPreset("Potato", "potato", "potato.yaml", "Potato_701",
                         "2002-04-15", "sowing", "2002-09-30", "maturity", .22, 197.9, 4930.,
                         "calibrated for northern Europe incl. Germany: sowing 15 April"),
    "sugar_beet": CropPreset("Sugar beet", "sugarbeet", "sugarbeet.yaml", "Sugarbeet_601",
                             "2002-04-05", "sowing", "2002-10-22", "harvest", .23, 42.0, 1620.,
                             "calibrated for Germany and neighbours: sowing 1-10 April, harvest 17-27 Oct"),
    "faba_bean": CropPreset("Faba bean", "fababean", "fababean.yaml", "Faba_bean_801",
                            "2002-03-15", "sowing", "2002-08-31", "maturity", .86, 250., 700., "EU15 set"),
    "soybean": CropPreset("Soybean", "soybean", "soybean.yaml", "Soybean_901",
                          "2002-04-28", "sowing", "2002-10-15", "maturity", .87, 450., 900.,
                          "calibrated for northern France: sowing 28 April, harvest about 25 Sep"),
    "winter_oilseed_rape": CropPreset("Winter oilseed rape", "rapeseed", "rapeseed.yaml", "Oilseed_rape_1001",
                                      "2002-01-01", "emergence", "2002-07-31", "maturity", .91, 450., 1100.,
                                      "western and northern Europe; temperature sums defined from 1 January"),
}


def crop_setup(name, engine_config, **crop_overrides):
    """CropConfig and EngineConfig for the selected preset.

    crop_overrides change CropConfig fields (e.g. variety, start, end); the economics come from the
    preset and can be changed afterwards with dataclasses.replace(engine_config, crop_price_eur_t=...)."""
    if name not in CROP_PRESETS:
        raise ValueError(f"Unknown crop '{name}'. Available: {sorted(CROP_PRESETS)}")
    p = CROP_PRESETS[name]
    fields = dict(variety=p.variety, start=p.start, end=p.end, start_type=p.start_type, crop_name=p.crop_name,
                  crop_file=p.crop_file, end_type=p.end_type, accept_demo_assumptions=True,
                  allow_missing_rain_for_pp=True)
    config = bridge.CropConfig(**{**fields, **crop_overrides})
    if config.variety not in bridge.crop_varieties(config.crop_file):
        raise ValueError(f"{config.variety} is not in {config.crop_file}: {sorted(bridge.crop_varieties(config.crop_file))}")
    engine_config = replace(engine_config, grain_dry_matter_fraction=p.dry_matter_fraction,
                            crop_price_eur_t=p.price_eur_t, crop_cost_eur_planted_ha=p.cost_eur_ha)
    return config, engine_config


def list_crops():
    """Table of the presets and of all crops in the library (for the notebook)."""
    import pandas as pd
    presets = pd.DataFrame([dict(key=k, **{f: getattr(p, f) for f in ("label", "crop_file", "variety", "start", "end",
                                                                       "end_type", "price_eur_t", "cost_eur_ha")})
                            for k, p in CROP_PRESETS.items()])
    library = sorted(f.name for f in bridge.CROP_LIBRARY.glob("*.yaml") if f.name != "crops.yaml")
    return presets, library
