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

import re
from pathlib import Path

VAR_NAME = "t2m"
TIME_NAME = "time"
LAT_NAME = "latitude"
LON_NAME = "longitude"
LEV_NAME = "level"
PRES_NAME = "pressure"


# Model-difference files (see visualization/modelDiff.py) end in this, and
# live in the minuend model's own directory, next to its output.
DIFF_SUFFIX = "_diff.nc"

# An auto-generated simulation name (see OutputParams.set_default_simulation_name):
# InferStudio_<Model1>_<Model2>_..._Year_Month_Day_hh:mm:ss
_AUTO_NAME_RE = re.compile(r"^InferStudio_.+_(\d{4}_\d{2}_\d{2}_\d{2}:\d{2}:\d{2})$")


def model_file_stem(simulation_name, model):
    """Base name for the files one model writes within a simulation.

    The simulation name lists every model in the run, but each model's
    files hold only that model's data, so they're named for it alone:
    InferStudio_Aurora_Pangu_<timestamp> gives InferStudio_Aurora_<timestamp>
    for Aurora's files. A custom name gets the model appended instead
    (my_run gives my_run_Aurora).
    """
    m = _AUTO_NAME_RE.match(simulation_name)
    if m:
        return f"InferStudio_{model}_{m.group(1)}"
    return f"{simulation_name}_{model}"


def diff_file_path(model_a_dir, model_a_name, model_b_name) -> Path:
    """Where the (A minus B) difference file lives, whether or not it exists
    yet: in A's own directory, named like A's output with "A_minus_B" in
    place of the model name, e.g.
    <suite>/Aurora/InferStudio_Aurora_minus_Pangu_<timestamp>_diff.nc
    """
    model_a_dir = Path(model_a_dir)
    stem = model_file_stem(model_a_dir.parent.name,
                           f"{model_a_name}_minus_{model_b_name}")
    return model_a_dir / f"{stem}{DIFF_SUFFIX}"


def resolve_nc_glob(model_dir) -> list:
    """Return the netCDF files to open for a model's output directory:
    the CF-compliant *_cf.nc file(s) if any exist, otherwise the raw *.nc
    file(s) (for older runs that predate CF conversion, or in case
    conversion failed for some reason). Difference files stored alongside
    the model's output are left out - they aren't that model's data.

    A path to a single file (a difference file) is returned as-is.
    """
    model_dir = Path(model_dir)
    if model_dir.is_file():
        return [str(model_dir)]
    files = sorted(p for p in model_dir.glob("*.nc")
                   if not p.name.endswith(DIFF_SUFFIX))
    cf = [p for p in files if p.name.endswith("_cf.nc")]
    return [str(p) for p in (cf or files)]
