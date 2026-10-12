"""Sampling and scalarisation, independent of Radiance/PCSE.

Every target hour remains in the output. Its estimated incident power is
sample_power * target_GHI/sample_GHI. This preserves unshaded horizontal
irradiation exactly, NOT shaded/PV energy: geometry-dependent ratios are
approximated and must be checked with full-hour runs.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "representative_hours"
    hours_per_day: int = 4
    days_per_month: int = 3


def make_sampling_plan(hour_starts, ghi, dhi, config):
    """Return target_hour, sample_hour, scale, target_day; never invent weather.

    Day representatives are nearest observed days to monthly irradiation
    quantiles. Assign days using normalised irradiation + diffuse fraction.
    Hour representatives are the highest-GHI hour in each daylight stratum.
    Night targets have no sample and scale zero. All weights are irradiance
    ratios (not an unweighted multiplication by 24 or 30).
    """
    modes = {"full", "representative_hours", "representative_days", "combined"}
    if config.mode not in modes:
        raise ValueError(f"Sampling mode must be one of {sorted(modes)}")
    for name in ("hours_per_day", "days_per_month"):
        v = getattr(config, name)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be a positive integer")
    idx = pd.DatetimeIndex(hour_starts)
    g, d = np.asarray(ghi, float), np.asarray(dhi, float)
    if len(idx) != len(g) or len(g) != len(d) or not idx.is_unique:
        raise ValueError("Weather lengths must agree and hours must be unique")
    if not np.isfinite(g).all() or not np.isfinite(d).all() or (g < 0).any() or (d < 0).any():
        raise ValueError("GHI/DHI must be finite and nonnegative")
    if len(idx) > 1 and not ((idx[1:] - idx[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError("Sampling requires consecutive standard-time hours")
    f = pd.DataFrame({"target_hour": idx, "ghi": g, "dhi": d,
                      "target_day": idx.date, "hour": idx.hour})
    daily = f.groupby("target_day").agg(ghi=("ghi", "sum"), dhi=("dhi", "sum"))
    daily["diffuse_fraction"] = daily.dhi / daily.ghi.where(daily.ghi > 0, 1)
    assignments = {day: day for day in daily.index}
    if config.mode in {"representative_days", "combined"}:
        for _, month in daily.groupby([pd.Index(daily.index).map(lambda x: x.year),
                                       pd.Index(daily.index).map(lambda x: x.month)]):
            positive = month[month.ghi > 0]
            if positive.empty:
                continue
            k = min(config.days_per_month, len(positive))
            sorted_days = positive.sort_values("ghi", kind="stable").index
            positions = np.floor((np.arange(k) + .5) * len(sorted_days) / k).astype(int)
            representatives = sorted_days[positions]
            vals = positive[["ghi", "diffuse_fraction"]]
            std = vals.std(ddof=0).replace(0, 1)
            features = (vals - vals.mean()) / std
            rep_features = features.loc[representatives].to_numpy()
            for day in positive.index:
                dist = np.square(rep_features - features.loc[day].to_numpy()).sum(axis=1)
                assignments[day] = representatives[int(np.argmin(dist))]
            # Keep representative dates as their own representatives.
            for day in representatives:
                assignments[day] = day
    hour_maps = {}
    for day, group in f.groupby("target_day", sort=True):
        light = group[group.ghi > 0]
        if light.empty:
            continue
        selected = {}
        if config.mode in {"representative_hours", "combined"}:
            for chunk in np.array_split(light.index.to_numpy(), min(config.hours_per_day, len(light))):
                sample = int(f.loc[chunk, "ghi"].idxmax())
                for target in chunk:
                    selected[int(f.loc[target, "hour"])] = sample
        else:
            selected = {int(row.hour): int(i) for i, row in light.iterrows()}
        hour_maps[day] = selected
    sample_indices = []
    for i, row in f.iterrows():
        if row.ghi <= 0:
            sample_indices.append(-1)
            continue
        day = assignments[row.target_day]
        available = hour_maps[day]
        h = int(row.hour)
        nearest = h if h in available else min(available, key=lambda x: (abs(x-h), x))
        sample_indices.append(available[nearest])
    samples = np.asarray(sample_indices)
    scales = np.zeros(len(f))
    valid = samples >= 0
    scales[valid] = g[valid] / g[samples[valid]]
    f["sample_hour"] = [idx[i] if i >= 0 else pd.NaT for i in samples]
    f["scale"] = scales
    return f[["target_hour", "target_day", "sample_hour", "scale"]]


@dataclass(frozen=True)
class PlotGrid:
    """Rotated row-aligned plot; the very same points go to ray tracing and crop."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    azimuth_deg: float
    z_m: float = .05

    @property
    def area_m2(self):
        return (self.x_max-self.x_min) * (self.y_max-self.y_min)

    def points(self):
        if self.nx < 1 or self.ny < 1 or self.area_m2 <= 0:
            raise ValueError("Invalid grid")
        dx, dy = (self.x_max-self.x_min)/self.nx, (self.y_max-self.y_min)/self.ny
        a = np.radians(self.azimuth_deg)
        ex, ey = np.array([-np.cos(a), np.sin(a)]), np.array([-np.sin(a), -np.cos(a)])
        records = []
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = (self.x_min+(i+.5)*dx)*ex + (self.y_min+(j+.5)*dy)*ey
                records.append(dict(point_id=f"p{i:03d}_{j:03d}", x_m=float(x), y_m=float(y), cell_area_m2=dx*dy))
        return pd.DataFrame(records)

    def scan(self):
        p = self.points().iloc[0]
        dx, dy = (self.x_max-self.x_min)/self.nx, (self.y_max-self.y_min)/self.ny
        a = np.radians(self.azimuth_deg)
        return dict(xstart=float(p.x_m), ystart=float(p.y_m), zstart=self.z_m,
                    sx_xinc=-np.cos(a)*dx, sx_yinc=np.sin(a)*dx, sx_zinc=0.,
                    xinc=-np.sin(a)*dy, yinc=-np.cos(a)*dy, zinc=0.,
                    Nx=self.nx, Ny=self.ny, Nz=1, orient="0 0 -1"), dx*dy


def resolve_array(array, design):
    """Array for ONE design. With a fixed field (array.field_depth_m > 0) the number
    of rows follows from the pitch: n_rows = floor(depth / pitch), at least 1.
    With field_depth_m == 0 (or absent) the array is returned unchanged (old behaviour)."""
    depth = float(getattr(array, "field_depth_m", 0.) or 0.)
    if depth <= 0:
        return array
    if not np.isfinite(design.pitch) or design.pitch <= 0:
        raise ValueError("pitch must be positive")
    return replace(array, n_rows=max(1, int(np.floor(depth/design.pitch + 1e-9))))


def field_length_m(array, design):
    """Length of the plot across the rows: the fixed field depth, or rows x pitch."""
    depth = float(getattr(array, "field_depth_m", 0.) or 0.)
    return depth if depth > 0 else array.n_rows*design.pitch


def design_grid(design, module, array, kind, nx=15, ny=15, z_m=.05):
    gap = design.xgap_conventional if kind == "conventional" else 0.
    width = array.n_mods*module.module_x + (array.n_mods-1)*gap
    length = field_length_m(array, design)
    return PlotGrid(-width/2, width/2, -length/2, length/2, nx, ny, design.azimuth, z_m)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def compare_sampling(approx, full):
    """Pair metrics by design AND construction; small full values need absolute errors."""
    metrics = ["electrical_kwh", "crop_yield_t_ha", "LER", "annual_net_eur_ha"]
    merged = approx.merge(full, on=["design_id", "pv_type"], suffixes=("_sampled", "_full"), validate="one_to_one")
    if len(merged) != len(approx) or len(merged) != len(full):
        raise ValueError("Approximate/full result sets must match")
    for c in metrics:
        diff = merged[c+"_sampled"] - merged[c+"_full"]
        merged[c+"_error"] = diff
        merged[c+"_error_pct"] = 100*diff/merged[c+"_full"].abs().replace(0, np.nan)
    return merged
