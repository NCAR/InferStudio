"""VAPOR's built-in colormaps, computed the way VAPOR computes them.

Each palette in static/palettes/<category>/<name>.tf3 is a handful of
control points - HSV and a position from 0 to 1 - and an interpolation
type. colormap() samples it with a port of VAPOR's
ColorMap::colorNormalized (lib/params/ColorMap.cpp) and the colour
conversions it uses (lib/params/TFInterpolator.cpp), quirks included, so
the colours match VAPOR's. Diverging maps use Kenneth Moreland's
interpolation through Msh space ("Diverging Color Maps for Scientific
Visualization", 2009; https://www.kennethmoreland.com/color-advice/), as
VAPOR and ParaView do.

Ported from VAPOR, Copyright (c) 2024, NSF National Center for
Atmospheric Research, under its BSD 3-Clause license
(static/palettes/LICENSE-VAPOR.txt).
"""

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

PALETTE_DIR = Path(__file__).resolve().parent.parent / "static" / "palettes"

# VAPOR's menu order: its categories and, within each, its files sorted
# by name, capitals first.
CATEGORIES = ("Diverging", "Highlighting", "Sequential")

# What a new plot starts with, and what difference panels always use:
# Moreland's cool-warm, VAPOR's (and ParaView's) default.
DEFAULT = "CoolWarm"
DIFF = "CoolWarm"

# Appended to a name for the same map reversed (VAPOR's Reverse Colormap).
REVERSED = "_r"

N_COLORS = 256

# TFInterpolator::type
LINEAR_HSV, DISCRETE, LOGARITHM, EXPONENTIAL, DIVERGING, LINEAR_RGB, LINEAR_LAB, LINEAR_LCH = range(8)

# --- TFInterpolator: colour conversions ----------------------------------- #

_XN, _YN, _ZN = 95.047, 100.0, 108.883
_XYZ = (0.4124564, 0.2126729, 0.0193339, 0.3575761, 0.7151522, 0.1191920,
        0.1804375, 0.0721750, 0.9503041)
_XYZ_INV = (3.2404548360214087, -0.9692663898756537, 0.05564341960421365,
            -1.5371388501025751, 1.876010928842491, -0.20402585426769812,
            -0.4985315468684809, 0.04155608234667354, 1.0572251624579287)


def _rgb2srgb(rgb):
    """Linear 0..100 to gamma-encoded 0..255."""
    out = []
    for c in rgb:
        v = c / 100.0
        v = 1.055 * v ** (1 / 2.4) - 0.055 if v > 0.00313080495356037152 else v * 12.92
        out.append(v * 255.0)
    return out


def _srgb2rgb(srgb):
    """Gamma-encoded 0..255 to linear 0..100."""
    out = []
    for c in srgb:
        v = c / 255.0
        v = ((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92
        out.append(v * 100)
    return out


def _srgb2xyz(srgb):
    r, g, b = _srgb2rgb(srgb)
    m = _XYZ
    return (r * m[0] + g * m[3] + b * m[6],
            r * m[1] + g * m[4] + b * m[7],
            r * m[2] + g * m[5] + b * m[8])


def _xyz2srgb(xyz):
    x, y, z = xyz
    m = _XYZ_INV
    return _rgb2srgb((x * m[0] + y * m[3] + z * m[6],
                      x * m[1] + y * m[4] + z * m[7],
                      x * m[2] + y * m[5] + z * m[8]))


def _lab_f(v):
    # A 32-bit float, as VAPOR's srgb2labHelper returns: subtracting the
    # 64-bit 16/116 from it then leaves black's L* at about -6e-8, not 0,
    # which is what keeps Moreland's Msh defined (and VAPOR's BlackBody as
    # it is) for black.
    return float(np.float32(v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116))


def _lab_finv(v):
    return v ** 3 if v > 7.787 * 0.008856 + 16 / 116 else (v - 16 / 116) / 7.787


def _srgb2lab(srgb):
    x, y, z = _srgb2xyz(srgb)
    fy = _lab_f(y / _YN)
    return (116 * (fy - 16 / 116), 500 * (_lab_f(x / _XN) - fy), 200 * (fy - _lab_f(z / _ZN)))


def _lab2srgb(lab):
    l, a, b = lab
    fy = (l + 16) / 116
    return _xyz2srgb((_XN * _lab_finv(a / 500 + fy), _YN * _lab_finv(fy),
                      _ZN * _lab_finv(fy - b / 200)))


def _lab2msh(lab):
    l, a, b = lab
    m = math.sqrt(l * l + a * a + b * b)
    return m, math.acos(l / m), math.atan2(b, a)


def _msh2lab(msh):
    m, s, h = msh
    return m * math.cos(s), m * math.sin(s) * math.cos(h), m * math.sin(s) * math.sin(h)


def _rgb2hsv(rgb):
    """rgb in any scale; h in degrees, s 0..1, v in rgb's scale."""
    lo, hi = min(rgb), max(rgb)
    delta = hi - lo
    if delta < 0.00001:
        return 0.0, 0.0, hi
    if hi <= 0:
        return math.nan, 0.0, hi
    r, g, b = rgb
    if r >= hi:
        h = (g - b) / delta
    elif g >= hi:
        h = 2 + (b - r) / delta
    else:
        h = 4 + (r - g) / delta
    h *= 60
    return (h + 360 if h < 0 else h), delta / hi, hi


def _hsv2rgb(h, s, v):
    """TFInterpolator::hsv2rgb: h 0..1."""
    if s <= 0:
        return v, v, v
    hh = h * 360.0
    if hh >= 360.0:
        hh = 0.0
    hh /= 60.0
    i = int(hh)
    ff = hh - i
    p, q, t = v * (1 - s), v * (1 - s * ff), v * (1 - s * (1 - ff))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v)][i] if i < 5 else (v, p, q)


def _color_to_rgb(h, s, v):
    """ColorMap::Color::toRGB: HSV (0..1) to display RGB (0..1)."""
    if s == 0:
        return v, v, v
    sector = int(h * 6)
    frac = h * 6 - sector
    if sector == 6:
        sector = 0
    a, b, c = v * (1 - s), v * (1 - frac * s), v * (1 - s * (1 - frac))
    return [(v, c, a), (b, v, a), (a, v, c), (a, b, v), (c, a, v), (v, a, b)][sector]


# --- Moreland's diverging interpolation ----------------------------------- #

def _adjust_hue(m, s, h, m_unsat):
    if m >= m_unsat - 0.1:
        return h
    spin = s * math.sqrt(m_unsat ** 2 - m ** 2) / (m * math.sin(s))
    return h + spin if h > -math.pi / 3 else h - spin


def _diverging(rgb1, rgb2, interp, corrective):
    """TFInterpolator::divergentInterpolation: rgb 0..255."""
    interp = 1 - interp
    m1, s1, h1 = _lab2msh(_srgb2lab(rgb1))
    m2, s2, h2 = _lab2msh(_srgb2lab(rgb2))
    # Two saturated, distinct colours: white (or the brighter end) between.
    if corrective and s1 > 0.05 and s2 > 0.05 and abs(h1 - h2) > 3.1415 / 3:
        if m1 > m2 and m1 > 88:
            mid = m1
        elif m2 > m1 and m2 > 88:
            mid = m2
        else:
            mid = 88.0
        if interp < 0.5:
            m2, s2, h2 = mid, 0.0, 0.0
            interp = 2 * interp
        else:
            m1, s1, h1 = mid, 0.0, 0.0
            interp = 2 * interp - 1
    if s1 < 0.05 and s2 > 0.05:
        h1 = _adjust_hue(m2, s2, h2, m1)
    elif s2 < 0.05 and s1 > 0.05:
        h2 = _adjust_hue(m1, s1, h1, m2)
    out = list(_lab2srgb(_msh2lab(((1 - interp) * m1 + interp * m2,
                                   (1 - interp) * s1 + interp * s2,
                                   (1 - interp) * h1 + interp * h2))))
    top = max(out)
    if top > 255:
        out = [c * 255 / top for c in out]
    return [max(c, 0.0) for c in out]


# --- ColorMap::colorNormalized --------------------------------------------- #

def _clamp3(c, lo, hi):
    return [min(max(v, a), b) for v, a, b in zip(c, lo, hi)]


def _hsv_out(rgb):
    h, s, v = _rgb2hsv(rgb)
    return h / 360, s, v


def _lab_of(hsv):
    rgb = [c * 100 for c in _color_to_rgb(*hsv)]
    return _clamp3(_srgb2lab(_rgb2srgb(rgb)), (0, -110, -110), (100, 110, 110))


def _from_lab(lab):
    rgb = [c / 100 for c in _srgb2rgb(_lab2srgb(lab))]
    return _hsv_out(_clamp3(rgb, (0, 0, 0), (1, 1, 1)))


def _lab2lch(lab):
    l, a, b = lab
    return l, math.sqrt(a * a + b * b), (math.atan2(b, a) / (2 * math.pi) * 360 + 360) % 360


def _lch2lab(lch):
    l, c, h = lch
    h = h / 360 * 2 * math.pi
    return l, math.cos(h) * c, math.sin(h) * c


def _interp_circ(a, b, r):
    if abs(b - a) <= 0.5:
        return a * (1 - r) + r * b
    v = (a + 1) * (1 - r) + r * b if a <= b else a * (1 - r) + r * (b + 1)
    return v - 1 if v >= 1 else v


def _segment_color(itype, whitespace, a, b, r):
    """HSV (0..1) at fraction r (0 < r < 1) between control points a, b."""
    if itype == DIVERGING:
        rgb1 = [c * 255 for c in _hsv2rgb(*a)]
        rgb2 = [c * 255 for c in _hsv2rgb(*b)]
        # VAPOR passes them right to left, and _diverging turns r around.
        h, s, v = _rgb2hsv(_diverging(rgb2, rgb1, r, whitespace))
        return h / 360, s, v / 255
    if itype == LINEAR_RGB:
        ca, cb = _color_to_rgb(*a), _color_to_rgb(*b)
        return _hsv_out([x + (y - x) * r for x, y in zip(ca, cb)])
    if itype == LINEAR_LAB:
        la, lb = _lab_of(a), _lab_of(b)
        return _from_lab([x + (y - x) * r for x, y in zip(la, lb)])
    if itype == LINEAR_LCH:
        ca = _clamp3(_lab2lch(_lab_of(a)), (0, 0, 0), (100, 140, 360))
        cb = _clamp3(_lab2lch(_lab_of(b)), (0, 0, 0), (100, 140, 360))
        ah, bh = ca[2], cb[2]
        if abs(bh - ah) > 180:
            if bh > ah:
                ah += 360
            else:
                bh += 360
        lch = (ca[0] + (cb[0] - ca[0]) * r, ca[1] + (cb[1] - ca[1]) * r, ah + (bh - ah) * r)
        return _from_lab(_clamp3(_lch2lab(lch), (0, -110, -110), (100, 110, 110)))
    if itype == DISCRETE:
        return a if r < 0.5 else b
    # Linear in HSV, the hue the short way round (logarithm and exponential
    # are never used for colour, and VAPOR falls back to this for them).
    return (_interp_circ(a[0], b[0], r), a[1] * (1 - r) + r * b[1], a[2] * (1 - r) + r * b[2])


def _color_at(points, itype, whitespace, nv):
    """ColorMap::colorNormalized: HSV at normalised value nv."""
    index = len(points) - 1
    for i, p in enumerate(points):
        if p[3] > nv:
            index = i - 1
            break
    if index < 0:
        return points[0][:3]
    if index >= len(points) - 1:
        return points[-1][:3]
    a, b = points[index], points[index + 1]
    r = (nv - a[3]) / (b[3] - a[3])
    if 0 < r < 1:
        return _segment_color(itype, whitespace, a[:3], b[:3], r)
    return b[:3] if r >= 1 else a[:3]


# --- palettes --------------------------------------------------------------- #

def _read_tf3(path):
    """(control points [(h, s, v, pos)], interpolation type, whitespace)."""
    text = Path(path).read_text()
    cm = text[text.index("<ColorMapParams>"):]

    def tag(name):
        m = re.search(rf"<{name}[^>]*>([^<]*)<", cm)
        return m.group(1).split() if m else None

    values = [float(v) for v in tag("ColorMapControlPoints")]
    points = [(min(max(values[i], 0.0), 1.0), values[i + 1], values[i + 2], values[i + 3])
              for i in range(0, len(values) - 3, 4)]
    whitespace = tag("UseWhitespace")
    return points, int(tag("ColorInterpolationType")[0]), (
        True if whitespace is None else bool(int(whitespace[0])))


@lru_cache(maxsize=None)
def palettes():
    """{category: [name, ...]} in VAPOR's menu order."""
    return {cat: sorted(p.stem for p in (PALETTE_DIR / cat).glob("*.tf3"))
            for cat in CATEGORIES}


def split_name(name):
    """(palette name, reversed?) for a name as colormap() takes it."""
    if name.endswith(REVERSED):
        return name[:-len(REVERSED)], True
    return name, False


@lru_cache(maxsize=None)
def category_of(name):
    """The category a palette (reversed or not) is in, or None."""
    base = split_name(name)[0]
    return next((cat for cat, names in palettes().items() if base in names), None)


@lru_cache(maxsize=None)
def colors(name, n=N_COLORS):
    """The palette `name` (or `<name>_r`, reversed) as n hex colours."""
    base, reverse = split_name(name)
    path = PALETTE_DIR / category_of(base) / f"{base}.tf3"
    table = path.with_suffix(".csv")
    if table.exists():
        rgbs = _from_table(table, n)
    else:
        points, itype, whitespace = _read_tf3(path)
        rgbs = [_color_to_rgb(*_color_at(points, itype, whitespace, i / (n - 1)))
                for i in range(n)]
    out = ["#{:02x}{:02x}{:02x}".format(*(int(round(min(max(c, 0.0), 1.0) * 255))
                                          for c in rgb))
           for rgb in rgbs]
    return out[::-1] if reverse else out


def _from_table(path, n):
    """n colours from a published colour table (scalar,r,g,b per line, from
    0 to 1), interpolated linearly between its entries.

    Used in place of the .tf3 where Kenneth Moreland's own table for a map
    is beside it (Kindlmann, KindlmannExtended, BlackBody): VAPOR's few
    control points, blended in HSV (or Msh for BlackBody), only roughly
    follow his maps, whose lightness rises evenly from end to end, and show
    bands where theirs dips or kinks. See static/palettes/README.md."""
    t = np.loadtxt(path, delimiter=",", skiprows=1)
    x = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x, t[:, 0], t[:, k]) for k in (1, 2, 3)], axis=1)


@lru_cache(maxsize=None)
def colormap(name):
    """The palette as a matplotlib colormap, for the plots and video
    export; HoloViews takes it as is."""
    return ListedColormap(colors(name), name=name)


def all_colormaps():
    """{name: colormap} for every palette, and every one reversed."""
    return {n: colormap(n) for names in palettes().values()
            for base in names for n in (base, base + REVERSED)}
