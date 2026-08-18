"""Naming-convention parsing utilities for earth2studio-format model
output (AIFS, Aurora, Pangu, and similar). Unlike ERA5 files, these
datasets do not store pressure-level fields as a single 3D variable with
a `level` dimension — each level is flattened into its own variable,
e.g. `u100`, `u150`, ..., `u1000`. Surface/single-level fields (t2m, msl,
sp, z500, ...) are stored as ordinary standalone variables.

Deliberately has NO dependency on xarray or matplotlib — this module
needs to be importable inside the lean, isolated per-model conda
environments (aifs/aurora/pangu) used for actual inference, which don't
have plotting libraries installed. earth2StudioPlot.py (the actual
plotting code, which does need matplotlib) imports from here rather than
duplicating this logic; cf_convert.py imports from here directly and
never touches earth2StudioPlot.py at all, for the same reason.
"""

import re
from pathlib import Path

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
