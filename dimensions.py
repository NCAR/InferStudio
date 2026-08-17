"""Shared NetCDF dimension/variable name conventions used across both the
Visualization and Inference tabs.

These are pure static string constants only — no filesystem access or
computed defaults. (era5_plot.py's NETCDF_FILE, by contrast, is computed
at import time and stays in visualization/era5_plot.py since it's specific
to that module's own plotting logic, not a shared naming convention.)
"""

VAR_NAME = "t2m"
TIME_NAME = "time"
LAT_NAME = "latitude"
LON_NAME = "longitude"
LEV_NAME = "level"
PRES_NAME = "pressure"
