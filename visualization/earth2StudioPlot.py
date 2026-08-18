"""Plotting utilities for earth2studio-format model outputs (AIFS, Aurora,
Pangu, and similar). See earth2StudioVars.py for the naming-convention
parsing logic (kept separate so code that only needs parsing, not
plotting, doesn't have to import matplotlib transitively).

Provides a plotting function that mirrors era5_plot.plot_png's interface
closely enough to slot into DatasetPlot2 in place of dummy_model_plot.
"""

import io
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dimensions import resolve_nc_glob, PRES_NAME
from visualization.earth2StudioVars import (
    LAT_NAMES, LON_NAMES, TIME_NAMES,
    _resolve_dim, parse_variable_groups, available_levels,
    resolve_var_name, get_model_nc_path,
)


def plot_e2s_field(model_dir, base_or_var, level, t, cmap="viridis", vmin=None, vmax=None):
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
    vmin, vmax : float or None
        Fixed colorbar range. If either is None, it's computed from this
        field's own actual data (matching matplotlib's default auto
        behavior), but explicitly, so the actual value used can be
        returned to the caller rather than staying hidden inside
        matplotlib's own internals.

    Returns
    -------
    (buf, vmin_used, vmax_used) : (io.BytesIO, float, float)
        The rendered PNG, and the actual min/max values used for the
        colorbar (whether passed in explicitly or auto-computed).
    """
    model_dir = Path(model_dir)

    with xr.open_mfdataset(resolve_nc_glob(model_dir), engine="netcdf4", autoclose=True, data_vars="all") as ds:
        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))
        var_name = resolve_var_name(level_vars, surface_vars, base_or_var, level)

        lat_dim = _resolve_dim(ds, *LAT_NAMES)
        lon_dim = _resolve_dim(ds, *LON_NAMES)

        da = ds[var_name]

        # CF-compliant files (post cf_convert.py) stack all pressure
        # levels of a variable into one array with a real `pressure`
        # dimension, rather than earth2studio's original flattened
        # per-level variables (u100, u850, ...). If this variable has a
        # real pressure dim, select the requested level from it directly
        # here — resolve_var_name won't have done this, since for a CF
        # file base_or_var already IS the actual variable name (no
        # flattened-name lookup needed).
        if PRES_NAME in da.dims and level is not None:
            da = da.sel({PRES_NAME: level}, method="nearest")

        # Capture the init/cycle time (if present) before it potentially
        # gets squeezed out below as a singleton dim — needed to compute
        # the actual valid time for the title even though this size-1 axis
        # isn't the one selected by `t`.
        init_time = ds["time"].values[0] if "time" in ds.coords else None

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

        select_dim = None
        t_clamped = t
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

        # Compute valid time / forecast (lead) time for the title, from
        # whichever coordinate we actually selected by, if possible.
        valid_time = None
        lead_hours = None
        if select_dim is not None and select_dim in ds.coords:
            coord_val = ds[select_dim].values[t_clamped]
            if np.issubdtype(ds[select_dim].dtype, np.timedelta64):
                # select_dim is a lead-time-style offset (e.g. "lead_time")
                lead_hours = coord_val / np.timedelta64(1, "h")
                if init_time is not None:
                    valid_time = init_time + coord_val
            elif np.issubdtype(ds[select_dim].dtype, np.datetime64):
                # select_dim IS itself the actual timestamp (ERA5-style)
                valid_time = coord_val
                first_val = ds[select_dim].values[0]
                lead_hours = (coord_val - first_val) / np.timedelta64(1, "h")

        data = da.values
        lats = ds[lat_dim].values if lat_dim else np.arange(data.shape[0])
        lons = ds[lon_dim].values if lon_dim else np.arange(data.shape[1])

    # Compute the actual min/max explicitly (rather than letting
    # matplotlib silently do it internally) so the values actually used
    # can be reported back to the caller for display.
    vmin_used = float(np.nanmin(data)) if vmin is None else vmin
    vmax_used = float(np.nanmax(data)) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(7, 3.8))
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, shading="auto", vmin=vmin_used, vmax=vmax_used)

    if valid_time is not None and lead_hours is not None:
        valid_str = str(np.datetime64(valid_time, "s"))
        title = f"{var_name}  |  Valid: {valid_str}  |  Forecast: +{lead_hours:.0f}h"
    elif lead_hours is not None:
        title = f"{var_name}  |  Forecast: +{lead_hours:.0f}h"
    else:
        title = f"{var_name}  (index={t})"

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf, vmin_used, vmax_used
