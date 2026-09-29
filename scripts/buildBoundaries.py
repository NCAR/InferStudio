#!/usr/bin/env python
"""Build the geographic boundary line files the plot grid overlays.

Downloads the Natural Earth 1:50m layers listed in
visualization/boundaries.py (LAYERS) through cartopy and writes each one to
static/boundaries/<layer>.npz as two float32 arrays, lon and lat, with NaN
between separate lines - the form Bokeh's line glyph draws directly.
Polygon layers (lakes) are reduced to their outlines.

The .npz files are committed, so the app itself never needs cartopy or
network access for this. Rerun only to change the layers or the scale.
Natural Earth is public domain (https://www.naturalearthdata.com/).

Usage:

    python scripts/buildBoundaries.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from visualization.boundaries import LAYERS, BOUNDARY_DIR, SCALE  # noqa: E402


def _lines(geom):
    """Yield each line of a (Multi)LineString or (Multi)Polygon as an
    (N, 2) coordinate array."""
    kind = geom.geom_type
    if kind == "LineString":
        yield np.asarray(geom.coords)
    elif kind == "Polygon":
        yield np.asarray(geom.exterior.coords)
        for ring in geom.interiors:
            yield np.asarray(ring.coords)
    elif kind.startswith("Multi") or kind == "GeometryCollection":
        for part in geom.geoms:
            yield from _lines(part)


def build_layer(layer):
    import cartopy.io.shapereader as shpreader

    category, name = LAYERS[layer]
    path = shpreader.natural_earth(resolution=SCALE, category=category,
                                   name=name)
    lon, lat = [], []
    for geom in shpreader.Reader(path).geometries():
        for coords in _lines(geom):
            if len(coords) < 2:
                continue
            lon += [coords[:, 0], [np.nan]]
            lat += [coords[:, 1], [np.nan]]
    lon = np.concatenate(lon).astype(np.float32)
    lat = np.concatenate(lat).astype(np.float32)
    out = BOUNDARY_DIR / f"{layer}.npz"
    np.savez_compressed(out, lon=lon, lat=lat)
    return out, int(np.isfinite(lon).sum())


def main():
    warnings.simplefilter("ignore")
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        out, n = build_layer(layer)
        print(f"{out.relative_to(REPO)}: {n} points, "
              f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
