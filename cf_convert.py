"""Convert earth2studio's flattened-variable NetCDF output (e.g. u100,
u850, ...) into a CF-compliant file with a genuine `pressure` dimension,
so cross-section (XZ/YZ) plotting has real vertical structure to slice
through instead of needing to stack many separate flattened variables at
plot time.

This only stacks variables InferStudio actually runs (matching the base
names in MODEL_VAR_MAP in earth2StudioRunner.py: u, v, t, q, z). Surface
only fields (t2m, sp, msl, ...) are copied through unchanged, aside from
CF metadata being stamped on if we recognize the variable name.

Deliberately keeps earth2studio's own lowercase variable-name convention
(u, v, t, q, z) rather than mimicking any uppercase convention seen in
other CF-compliant reference files elsewhere — the rest of InferStudio's
codebase (parse_variable_groups, resolve_var_name, the Variable dropdown)
is built entirely around lowercase names already.
"""

from pathlib import Path

import numpy as np
import xarray as xr

# Reuse the SAME level-parsing/dim-resolution logic already used for
# plotting, so this stays in sync automatically rather than duplicating
# the flattened-variable-name regex or lat/lon-dimension detection.
from visualization.earth2StudioVars import parse_variable_groups, _resolve_dim, LAT_NAMES, LON_NAMES

# CF metadata for the base variables InferStudio actually runs. Extend
# this if new base variables are added to MODEL_VAR_MAP in
# earth2StudioRunner.py.
_CF_VAR_ATTRS = {
    "u": {"standard_name": "eastward_wind", "long_name": "U component of wind", "units": "m s**-1"},
    "v": {"standard_name": "northward_wind", "long_name": "V component of wind", "units": "m s**-1"},
    "t": {"standard_name": "air_temperature", "long_name": "Temperature", "units": "K"},
    "q": {"standard_name": "specific_humidity", "long_name": "Specific humidity", "units": "kg kg**-1"},
    "z": {"standard_name": "geopotential", "long_name": "Geopotential", "units": "m**2 s**-2"},
}

# CF metadata for known surface/single-level fields, applied only if the
# variable is actually present in a given file.
_CF_SURFACE_ATTRS = {
    "t2m": {"standard_name": "air_temperature", "long_name": "2 meter temperature", "units": "K"},
    "sp": {"standard_name": "surface_air_pressure", "long_name": "Surface pressure", "units": "Pa"},
    "msl": {"standard_name": "air_pressure_at_mean_sea_level", "long_name": "Mean sea level pressure", "units": "Pa"},
}

_COORD_CF_ATTRS = {
    "latitude": {"long_name": "latitude", "short_name": "lat", "units": "degree_north", "axis": "Y"},
    "lat":      {"long_name": "latitude", "short_name": "lat", "units": "degree_north", "axis": "Y"},
    "longitude": {"long_name": "longitude", "short_name": "lon", "units": "degree_east", "axis": "X"},
    "lon":       {"long_name": "longitude", "short_name": "lon", "units": "degree_east", "axis": "X"},
}


def make_cf_compliant(nc_path: str, suffix: str = "_cf") -> str:
    """Read an earth2studio-format NetCDF file at `nc_path` and write a
    CF-compliant version alongside it (same directory, `<stem><suffix>.nc`).
    Returns the path to the new file. The original file is left untouched.
    """
    nc_path = str(nc_path)
    out_path = nc_path[:-3] + f"{suffix}.nc" if nc_path.endswith(".nc") else nc_path + suffix

    with xr.open_dataset(nc_path) as ds:
        level_vars, surface_vars = parse_variable_groups(list(ds.data_vars))

        new_vars = {}

        # Stack each leveled base variable (u, v, t, q, z, ...) into one
        # array with a genuine `pressure` dimension.
        for base, levels_map in level_vars.items():
            levels_sorted = sorted(levels_map.keys())
            pieces = [ds[levels_map[lev]] for lev in levels_sorted]

            # Determine lat/lon dim names and every other ("leading",
            # e.g. time/lead_time) dim from the first piece — should be
            # consistent across all levels of the same base variable.
            lat_dim = _resolve_dim(pieces[0], *LAT_NAMES)
            lon_dim = _resolve_dim(pieces[0], *LON_NAMES)
            leading_dims = [d for d in pieces[0].dims if d not in (lat_dim, lon_dim)]

            stacked = xr.concat(pieces, dim="pressure")
            stacked = stacked.assign_coords(
                pressure=("pressure", np.array(levels_sorted, dtype="float64"))
            )
            # xr.concat prepends the new dim — reorder so leading dims
            # (time, lead_time, ...) stay first, then pressure, then lat/lon.
            stacked = stacked.transpose(*leading_dims, "pressure", lat_dim, lon_dim)

            attrs = dict(_CF_VAR_ATTRS.get(base, {}))
            attrs.setdefault("short_name", base)
            stacked.attrs = attrs

            new_vars[base] = stacked

        # Surface/single-level fields pass through unchanged, but get CF
        # attrs stamped on if we recognize the variable name.
        for name in surface_vars:
            da = ds[name].copy()
            if name in _CF_SURFACE_ATTRS:
                da.attrs = {**da.attrs, **_CF_SURFACE_ATTRS[name]}
            new_vars[name] = da

        out_ds = xr.Dataset(new_vars, coords=ds.coords)

        # CF metadata for the new pressure coordinate.
        if "pressure" in out_ds.coords:
            out_ds["pressure"].attrs = {
                "long_name": "pressure",
                "short_name": "pres",
                "units": "hPa",
                "axis": "Z",
                "positive": "down",  # CF convention: pressure decreases upward
            }

        # Stamp CF metadata on lat/lon coords if not already present.
        for coord_name, cf_attrs in _COORD_CF_ATTRS.items():
            if coord_name in out_ds.coords and not out_ds.coords[coord_name].attrs:
                out_ds.coords[coord_name].attrs = dict(cf_attrs)

        out_ds.attrs["Conventions"] = "CF-1.11"

        out_ds.to_netcdf(out_path)

    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cf_convert.py <path_to_netcdf>")
        sys.exit(1)
    result = make_cf_compliant(sys.argv[1])
    print(f"Wrote CF-compliant file: {result}")
