"""Natural Earth boundary lines for overlaying on the plot grid's maps.

The lines are prebuilt by scripts/buildBoundaries.py into
static/boundaries/<layer>.npz (lon/lat float32 arrays, NaN between lines),
so loading one is a quick file read with no cartopy or network involved.
Natural Earth is public domain (https://www.naturalearthdata.com/).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

BOUNDARY_DIR = Path(__file__).resolve().parent.parent / "static" / "boundaries"

# 1:50m - detailed enough for a 0.25 degree grid without the ten-fold point
# count of 1:10m.
SCALE = "50m"

# layer -> (Natural Earth category, Natural Earth name)
LAYERS = {
    "coastline": ("physical", "coastline"),
    "countries": ("cultural", "admin_0_boundary_lines_land"),
    "states": ("cultural", "admin_1_states_provinces_lines"),
    "lakes": ("physical", "lakes"),
    "rivers": ("physical", "rivers_lake_centerlines"),
}

NONE = "None"
COASTLINES = "Coastlines"

# What a new plot opens with.
DEFAULT = COASTLINES

# Line style, shared by the plot grid (Bokeh, widths in screen pixels) and
# the video export's matplotlib frames (widths in points): a thin dark line
# over a wider, translucent light halo. Either colour alone vanishes
# somewhere - dark lines over viridis's dark purple, light ones over
# coolwarm's white middle - while the pair reads over any colormap without
# hiding the field.
BOUNDARY_COLOR = "#111111"
BOUNDARY_HALO_COLOR = "#ffffff"
BOUNDARY_HALO_ALPHA = 0.55
BOUNDARY_WIDTH = 0.8
BOUNDARY_HALO_WIDTH = 2.2
# The export's frames are 7in wide at 130 dpi, so a point is ~1.8 px; these
# land close to the on-screen pixel widths above.
MPL_BOUNDARY_WIDTH = 0.45
MPL_BOUNDARY_HALO_WIDTH = 1.2

# Dropdown label -> layers drawn. Coastlines are in every option: country or
# river lines floating without the land outline are hard to read. States and
# provinces at 1:50m cover the larger countries only (the US, Canada,
# Australia, Brazil, China, India, Russia and a few others).
BOUNDARY_OPTIONS = {
    NONE: (),
    COASTLINES: ("coastline",),
    "Coastlines + countries": ("coastline", "countries"),
    "Coastlines + countries + states/provinces":
        ("coastline", "countries", "states"),
    "Coastlines + lakes & rivers": ("coastline", "lakes", "rivers"),
}


@lru_cache(maxsize=None)
def _layer(layer):
    with np.load(BOUNDARY_DIR / f"{layer}.npz") as f:
        return f["lon"], f["lat"]


def _to_lon360(lon, lat):
    """Shift -180..180 lines onto a 0..360 map.

    Moving the western hemisphere up by 360 turns every line that crossed
    the prime meridian into one that jumps from ~360 back to ~0, which a
    line glyph would draw as a stroke straight across the map. A NaN is
    inserted at each such jump to break the line there instead.
    """
    lon = np.where(lon < 0, lon + 360, lon)
    jump = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1
    return np.insert(lon, jump, np.nan), np.insert(lat, jump, np.nan)


@lru_cache(maxsize=None)
def boundary_lines(option, lon360):
    """(lon, lat) float32 arrays for a BOUNDARY_OPTIONS label, NaN-separated.

    lon360 selects the longitude convention of the map they're drawn on:
    True for 0..360, False for -180..180. Unknown labels and "None" give
    empty arrays.
    """
    layers = BOUNDARY_OPTIONS.get(option, ())
    if not layers:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    parts = [_layer(layer) for layer in layers]
    lon = np.concatenate([p[0] for p in parts])
    lat = np.concatenate([p[1] for p in parts])
    if lon360:
        lon, lat = _to_lon360(lon, lat)
    return lon.astype(np.float32), lat.astype(np.float32)
