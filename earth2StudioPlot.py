"""Plotting utilities for earth2studio-format model outputs (AIFS, Aurora,
Pangu, and similar). Unlike ERA5 files, these datasets do not store
pressure-level fields as a single 3D variable with a `level` dimension —
each level is flattened into its own variable, e.g. `u100`, `u150`, ...,
`u1000`. Surface/single-level fields (t2m, msl, sp, z500, ...) are stored
as ordinary standalone variables.

This module parses that naming convention so a "base variable + level"
selection (e.g. base="u", level=850) can be resolved to the actual
variable name (`u850`) in the file, and provides a plotting function
that mirrors era5_plot.plot_png's interface closely enough to slot into
DatasetPlot2 in place of dummy_model_plot.
"""

import io
import re
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same fallback names used elsewhere in the codebase (see app_layout._resolve_dim)
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")
TIME_NAMES = ("time",)

_LEVEL_VAR_RE = re.compile(r"^([a-zA-Z]+?)(\d+)$")


def _resolve_dim(ds, *candidates):
    for name in candidates:
        if name in ds.sizes:
            return name
    return None


def parse_variable_groups(var_names):
    """Split a flat list of variable names into:
      - level_vars: {base_name: {level(int): full_var_name}}, e.g. {"u": {100: "u100", 850: "u850"}}
      - surface_vars: [names with no trailing numeric level, e.g. "t2m", "msl", "sp"]

    Note: t2m is NOT treated as a leveled variable — the regex requires the
    string to END in digits, and "t2m" ends in "m", so it correctly falls
    through to surface_vars.
    """
    level_vars = {}
    surface_vars = []
    for name in var_names:
        m = _LEVEL_VAR_RE.match(name)
        if m:
            base, lev = m.group(1), int(m.group(2))
            level_vars.setdefault(base, {})[lev] = name
        else:
            surface_vars.append(name)
    return level_vars, surface_vars


def available_levels(level_vars, base):
    """Sorted list of integer levels available for a given base variable name."""
    return sorted(level_vars.get(base, {}).keys())


def resolve_var_name(level_vars, surface_vars, base_or_var, level=None):
    """Resolve a (base, level) or plain surface variable name to the actual
    variable name present in the dataset.

    - If `base_or_var` is a known surface variable, `level` is ignored.
    - If `base_or_var` is a known leveled base name, `level` must match one
      of its available levels (nearest level is used if an exact match
      isn't found, so a slider index off by one doesn't hard-crash).
    """
    if base_or_var in surface_vars:
        return base_or_var

    levels = level_vars.get(base_or_var)
    if levels is None:
        raise KeyError(f"Unknown variable/base: {base_or_var!r}")

    if level in levels:
        return levels[level]

    # Fall back to nearest available level rather than raising
    nearest = min(levels.keys(), key=lambda l: abs(l - (level if level is not None else 0)))
    return levels[nearest]


def get_model_nc_path(model_dir) -> Path:
    """Return the model's output directory as a Path (xr.open_mfdataset glob
    target). Each model writes one or more .nc files per model directory."""
    return Path(model_dir)


def plot_e2s_field(model_dir, base_or_var, level, t, cmap="viridis") -> io.BytesIO:
    """Open the model's NetCDF output and render a single lat/lon field.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the model's .nc file(s), e.g. <sim_dir>/AIFS
    base_or_var : str
        Either a surface variable name (t2m, msl, sp, ...) or a leveled
        base name (u, v, t, q, z, ...)
    level : int or None
        Pressure level in hPa, used only when base_or_var is a leveled base.
    t : int
        Time index to select.
    """
    model_dir = Path(model_dir)

    with xr.open_mfdataset(f"{model_dir}/*.nc", engine="netcdf4", autoclose=True, data_vars="all") as ds:
        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
        var_name = resolve_var_name(level_vars, surface_vars, base_or_var, level)

        time_dim = _resolve_dim(ds, *TIME_NAMES)  # used only for the plot title, not selection
        lat_dim = _resolve_dim(ds, *LAT_NAMES)
        lon_dim = _resolve_dim(ds, *LON_NAMES)

        da = ds[var_name]

        # Don't rely on the axis actually being named "time" — earth2studio
        # outputs may call it lead_time/step/forecast_time/etc., and there
        # can be more than one extra axis (e.g. a size-1 "time" alongside a
        # size-7 "lead_time"). Squeeze singleton dims first, so whatever's
        # left over is the real forecast-step axis to select by `t`.
        extra_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]

        for d in list(extra_dims):
            if da.sizes[d] == 1:
                da = da.isel({d: 0})
                extra_dims.remove(d)

        if len(extra_dims) == 1:
            select_dim = extra_dims[0]
            t_clamped = min(max(t, 0), da.sizes[select_dim] - 1)
            da = da.isel({select_dim: t_clamped})
        elif len(extra_dims) > 1:
            raise ValueError(
                f"Variable {var_name!r} has multiple non-singleton extra "
                f"dimensions {extra_dims!r} beyond lat/lon — can't tell which "
                f"one the time index should select."
            )

        data = da.values
        lats = ds[lat_dim].values if lat_dim else np.arange(data.shape[0])
        lons = ds[lon_dim].values if lon_dim else np.arange(data.shape[1])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, shading="auto")
    title = f"{var_name}  (index={t})"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf
