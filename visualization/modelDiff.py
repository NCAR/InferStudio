"""Compute per-variable differences between two models' CF-compliant
output within the same simulation suite, writing the result to a new
NetCDF file that can be plotted the same way as any other model output
(via plot_e2s_field), just with a diverging colormap and a symmetric
value range, since these are signed difference fields.

Cached: once computed for a given model pair, the same file is reused on
subsequent requests rather than recomputing.

Each model pair gets its own subdirectory (cache_dir/<A>_minus_<B>/) —
matching the same "model_dir contains this model's .nc file(s)"
convention used everywhere else in the app (e.g. <sim_dir>/AIFS/AIFS.nc).
plot_e2s_field always globs *.nc within whatever directory it's given, so
if multiple diff pairs shared one flat directory, computing a second pair
for the same suite would make that glob match both files at once and try
to merge them together — giving each pair its own directory avoids that
entirely.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from dimensions import resolve_nc_glob
from visualization.earth2StudioVars import _resolve_dim, LAT_NAMES, LON_NAMES


def compute_model_difference(model_a_dir, model_b_dir, cache_dir, model_a_name, model_b_name) -> Path:
    """Compute (model_a - model_b) for every variable present in both
    datasets, writing the result to
    <cache_dir>/<model_a_name>_minus_<model_b_name>/<model_a_name>_minus_<model_b_name>.nc.

    Returns the PATH TO THE DIRECTORY containing that file (not the file
    itself) — this is the same "model_dir" contract plot_e2s_field
    expects for every other model. If the file already exists, its
    directory is returned immediately without recomputing.
    """
    cache_dir = Path(cache_dir)
    pair_name = f"{model_a_name}_minus_{model_b_name}"
    pair_dir = cache_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    out_path = pair_dir / f"{pair_name}.nc"

    if out_path.exists():
        return pair_dir

    with xr.open_mfdataset(resolve_nc_glob(model_a_dir), engine="netcdf4", autoclose=True, data_vars="all") as ds_a, \
         xr.open_mfdataset(resolve_nc_glob(model_b_dir), engine="netcdf4", autoclose=True, data_vars="all") as ds_b:

        shared_vars = sorted(set(ds_a.data_vars) & set(ds_b.data_vars))
        if not shared_vars:
            raise ValueError(
                f"{model_a_name} and {model_b_name} have no variables in "
                f"common \u2014 cannot compute a difference."
            )

        diff_vars = {}
        skipped = {}
        for var in shared_vars:
            da_a = ds_a[var]
            da_b = ds_b[var]
            try:
                if da_a.shape != da_b.shape:
                    # Grids don't line up (e.g. AIFS's 721-point latitude
                    # grid vs Aurora's 720-point grid). Interpolate ONLY
                    # the specific dimensions that actually mismatch, by
                    # name — NOT a blanket da_b.interp_like(da_a), which
                    # tries to interpolate every matching coordinate
                    # (including size-1 dims like "time"). Scipy's linear
                    # interpolator needs >=2 points to compute a slope; a
                    # size-1 axis hits an exact 0/0 division, and that
                    # single NaN then propagates through the ENTIRE array
                    # via broadcasting — silently turning a minor,
                    # legitimate lat-grid mismatch into 100% NaN output.
                    lat_dim = _resolve_dim(da_a, *LAT_NAMES)
                    lon_dim = _resolve_dim(da_a, *LON_NAMES)
                    interp_kwargs = {}
                    for dim in (lat_dim, lon_dim):
                        if (
                            dim
                            and dim in da_b.dims
                            and dim in da_a.dims
                            and (
                                da_a.sizes[dim] != da_b.sizes[dim]
                                or not np.array_equal(da_a[dim].values, da_b[dim].values)
                            )
                        ):
                            interp_kwargs[dim] = da_a[dim]
                    if interp_kwargs:
                        da_b = da_b.interp(**interp_kwargs)
                diff = da_a - da_b
                diff.attrs = dict(da_a.attrs)
                diff_vars[var] = diff
            except Exception as e:
                skipped[var] = str(e)

        if not diff_vars:
            raise ValueError(
                f"Could not compute a difference for any shared variable "
                f"between {model_a_name} and {model_b_name}: {skipped}"
            )

        out_ds = xr.Dataset(diff_vars)
        out_ds.attrs["Conventions"] = ds_a.attrs.get("Conventions", "CF-1.11")
        out_ds.attrs["history"] = f"Difference: {model_a_name} minus {model_b_name}"

        out_ds.to_netcdf(out_path)

    return pair_dir
