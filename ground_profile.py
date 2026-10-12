"""Ground light along the field depth for cached designs (diagnostic, no simulation).

Reads the cached Radiance ground results (ground_<construction>.csv.gz + manifest.json) of
designs and plots the crop-season mean daily light on the ground (Wh/m2/day) against the
position along the field depth (the direction across the rows).

Use it in a notebook cell:
    from ground_profile import profile_report
    profile_report(r"results\cache_v5",
                   {"pitch 4.05": "9594c193ca2b", "pitch 4.30": "c373b42ecc81", "pitch 4.80": "2f472ca55fe5"})
The values are cache-folder prefixes (the 12 characters printed as 'Physics cache hit: ...').
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def _folder(cache, prefix):
    hits = [p for p in Path(cache).glob(prefix + "*") if p.is_dir() and (p / "manifest.json").exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"{prefix}: expected exactly one cache folder, found {len(hits)}")
    return hits[0]


def depth_profile(cache, prefix, kind="conventional"):
    """Mean daily ground light per depth position; returns (DataFrame, manifest)."""
    folder = _folder(cache, prefix)
    manifest = json.loads((folder / "manifest.json").read_text())
    grid = manifest["grids"][kind]
    ground = pd.read_csv(folder / f"ground_{kind}.csv.gz")
    ground["j"] = ground.point_id.str.split("_").str[1].astype(int)
    ny = int(grid["ny"])
    depth = grid["y_max"] - grid["y_min"]
    # per-point crop-season mean first (same weight for every day), then mean over the width
    per_point = ground.groupby(["point_id", "j"], as_index=False).total_Wh_m2_day.mean()
    prof = per_point.groupby("j", as_index=False).total_Wh_m2_day.mean()
    prof["depth_m"] = grid["y_min"] + (prof.j + .5) * depth / ny
    prof = prof.rename(columns={"total_Wh_m2_day": "ground_Wh_m2_day"})
    return prof[["depth_m", "ground_Wh_m2_day"]], manifest


def profile_report(cache, designs, kind="conventional", save="ground_profile.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rows = []
    for label, prefix in designs.items():
        prof, manifest = depth_profile(cache, prefix, kind)
        d = manifest["design"]
        grid = manifest["grids"][kind]
        depth = grid["y_max"] - grid["y_min"]
        n_rows = max(1, int(np.floor(depth / d["pitch"] + 1e-9)))
        line, = ax.plot(prof.depth_m, prof.ground_Wh_m2_day, marker="o", ms=3, label=f"{label} ({n_rows} rows)")
        # ASSUMED row positions: rows centred on the field middle, one every `pitch` metres
        for k in range(n_rows):
            ax.axvline((k - (n_rows - 1) / 2) * d["pitch"], color=line.get_color(), ls=":", lw=.8, alpha=.7)
        outer = prof.depth_m.abs() > (prof.depth_m.abs().max() - d["pitch"])
        rows.append(dict(design=label, pitch=round(d["pitch"], 2), n_rows=n_rows,
                         mean_all=prof.ground_Wh_m2_day.mean(),
                         mean_inner=prof.ground_Wh_m2_day[~outer].mean(),
                         mean_outer_strips=prof.ground_Wh_m2_day[outer].mean(),
                         first_end=prof.ground_Wh_m2_day.iloc[0], last_end=prof.ground_Wh_m2_day.iloc[-1]))
    ax.set_xlabel("position along the field depth (m); dotted = ASSUMED row positions (centred)")
    ax.set_ylabel("crop-season mean ground light (Wh/m2/day)")
    ax.set_title(f"Ground light across the field ({kind})")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    table = pd.DataFrame(rows).round(2)
    print(table.to_string(index=False))
    print("Saved", save)
    return table
