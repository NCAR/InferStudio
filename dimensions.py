"""Shared NetCDF dimension/variable name conventions used across both the
Visualization and Inference tabs.

These are pure static string constants only — no filesystem access or
computed defaults. (era5_plot.py's NETCDF_FILE, by contrast, is computed
at import time and stays in visualization/era5_plot.py since it's specific
to that module's own plotting logic, not a shared naming convention.)

This module is a leaf — it must not import from cf_convert.py or
visualization/earth2StudioPlot.py, since both of those import from here
(directly or indirectly) and doing so would create a circular import.
"""

from pathlib import Path

VAR_NAME = "t2m"
TIME_NAME = "time"
LAT_NAME = "latitude"
LON_NAME = "longitude"
LEV_NAME = "level"
PRES_NAME = "pressure"


def resolve_nc_glob(model_dir) -> str:
    """Return the netCDF glob pattern to use for a model's output
    directory: prefer the CF-compliant *_cf.nc file if one exists,
    otherwise fall back to the raw *.nc file (for older runs that predate
    CF conversion, or in case conversion failed for some reason).
    """
    model_dir = Path(model_dir)
    if list(model_dir.glob("*_cf.nc")):
        return f"{model_dir}/*_cf.nc"
    return f"{model_dir}/*.nc"
