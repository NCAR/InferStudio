"""Linked, zoomable plot grid for InferStudio.

Replaces the per-model Matplotlib cards with a single HoloViews Layout whose
panels share axes, so a zoom on any panel applies to all of them. Panels are
datashader-rasterized, so zooming re-aggregates server-side and resolves
finer structure rather than magnifying pixels.

Layout is (model, model-minus-other) per row:

    +----------------+  +--------------------------+
    | AIFS           |  | AIFS minus Aurora        |
    +----------------+  +--------------------------+
    | Aurora         |  | Aurora minus Pangu       |
    +----------------+  +--------------------------+

Navigation: scroll to zoom (over the map - scrolling over an axis does not
zoom that axis alone), drag to pan, double-click to reset. Dragging a value
on a colorbar rescales that column's colour range - see
_COLORBAR_DRAG_JS - and double-clicking a colorbar resets that range, where
double-clicking the map resets only the zoom. The first three
are permanently on and there is no toolbar to switch them off - the tools
are declared in _panel_opts and the toolbar is suppressed in layout(), two
different places for the reason the comment in layout() explains.

Panels are linked by HoloViews' shared_axes. Note this links them PAIRWISE
rather than all-to-all: instrumentation showed four figures carrying only
two distinct Range1d objects, so a zoom propagates within a pair but not
across the grid. Attempts to link all four explicitly - sharing Range1d
objects server-side, and CustomJS callbacks client-side - both failed, so
the partial linking stands for now. See the standalone repro script if
picking this up again.

Panels are FIXED-SIZE (see PANEL_FRAME_WIDTH): a narrow browser window
scrolls the grid horizontally rather than shrinking the maps.

Instead of Bokeh's per-panel hover tooltip, a single readout above the grid
reports every model's value at the cursor position simultaneously - which is
the comparison the grid exists to support, and which a tooltip showing one
panel at a time cannot give. See _on_pointer.

The per-cell "Compute Difference" selectors moved to the sidebar (see
PlotGrid.diff_selectors) - Panel widgets can't be interleaved inside an
hv.Layout, and the sidebar is where the other field controls already live.

Typical use:

    grid = PlotGrid(models, model_dirs, state=shared_state)
    grid.set_time_bounds(n_steps)
    sidebar.append(grid.diff_selectors())
    plots_card = grid.card(title=suite_name)
    grid.refresh_clims_async()

    ...later, when the suite changes:
    grid.teardown()
"""

from __future__ import annotations

import threading
import time
import traceback
from functools import partial
from html import escape
from pathlib import Path

import numpy as np
import param
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import rasterize
from bokeh.models import (
    ColumnDataSource, CustomJS, CustomJSTickFormatter, FixedTicker,
    WheelZoomTool)

from visualization.earth2StudioPlot import (
    load_e2s_field, field_range, FieldReadError, CANON_LAT, CANON_LON)
from visualization.boundaries import (
    boundary_lines, DEFAULT as BOUNDARIES_DEFAULT, BOUNDARY_COLOR, BOUNDARY_WIDTH,
    BOUNDARY_HALO_COLOR, BOUNDARY_HALO_WIDTH, BOUNDARY_HALO_ALPHA)
from visualization.modelDiff import (
    load_diff_field,
    compute_model_difference,
    symmetric_diff_range,
)
from dimensions import diff_file_path
from cf_convert import variable_long_name

hv.extension("bokeh")


# Panel geometry.
#
# Panels are fixed-size rather than responsive. Responsive panels shrank to
# whatever width the window left over after the sidebar, which made the maps
# tiny and - because Bokeh sizes the colorbar and axis labels OUTSIDE the
# responsive box - clipped the colorbar annotations and the right-hand
# column. Fixed frames keep the maps legible; the main area scrolls
# horizontally (see app_layout's `main`) when they don't fit.
#
# frame_width/frame_height size the data area only, so the colorbar, its
# label and the axes are added around it at their natural size and are
# never squeezed. The 2:1 ratio matches GLOBAL_EXTENT. This sets the
# plot's rendered shape, not data_aspect, which would constrain the axis
# RANGES and fight zooming into any region that isn't 2:1.
ASPECT = 2.0
PANEL_FRAME_WIDTH = 560
PANEL_FRAME_HEIGHT = int(PANEL_FRAME_WIDTH / ASPECT)
COLORBAR_WIDTH = 12

# PANEL_FRAME_WIDTH is the smallest a panel gets, not its only size: in the
# browser, app_layout's PlotGridAutoSize widens every figure tagged
# AUTOSIZE_TAG to fill the space the window leaves, up to
# MAX_PANEL_FRAME_WIDTH, keeping ASPECT. Still fixed-size figures, just
# resized from outside - see PlotGridAutoSize for why that's not Bokeh's own
# responsive sizing.
MAX_PANEL_FRAME_WIDTH = 1600
AUTOSIZE_TAG = "plotgrid-autosize"

DEFAULT_CMAP = "viridis"
DIFF_CMAP = "coolwarm"

# Colormaps for panels carrying no data. Grey for "nothing selected", a
# tinted one for "computing" - the two states must not look alike, or a
# panel waiting on a slow difference computation reads as broken. See
# _placeholder.
PLACEHOLDER_CMAP = "gray"
COMPUTING_CMAP = "Blues"

# Minimum seconds between readout updates. PointerXY fires on every mouse
# move - without a gate that is hundreds of websocket messages per second of
# cursor travel, each one re-rendering an HTML pane. 25 Hz is smooth to the
# eye and roughly an order of magnitude less traffic.
POINTER_MIN_INTERVAL = 0.04

# Fallback extent for placeholder panels drawn before any real field has been
# loaded. These models write longitude on 0..360, not -180..180 - a
# placeholder claiming the wrong range would drag every panel to it once the
# axes are linked. __init__ seeds _last_extent from a real field, so this is
# only a fallback if that read fails.
GLOBAL_EXTENT = (0.0, -90.0, 360.0, 90.0)

# Tightest window a panel's wheel-zoom is allowed to reach, in degrees - a
# floor on how far in a user can scroll before the field turns into a few
# blocky rasterized pixels. Kept at the same 2:1 ratio as ASPECT/
# GLOBAL_EXTENT so the minimum view isn't a differently-shaped sliver of the
# full-map aspect.
MIN_LON_SPAN = 10.0
MIN_LAT_SPAN = 5.0

# Pan-only clamp: translates (never resizes) [start, end] back inside
# [lo, hi] if a drag has pushed it past either edge. Deliberately does NOT
# also enforce min_span here - recomputing an exact recenter on every
# start/end change fought WheelZoomTool's own incremental zoom math once the
# range was already at the floor, and float rounding across repeated
# corrections made both zoom and pan flicker/stick at that point. The span
# floor/ceiling is left entirely to Range1d.min_interval/max_interval (see
# _apply_zoom_clamp) instead, which isn't implicated in the bug below.
_PAN_CLAMP_JS = """
let start = range.start
let end = range.end
if (start < lo) { end += (lo - start); start = lo }
if (end > hi) { start -= (end - hi); end = hi }
start = Math.max(start, lo)
end = Math.min(end, hi)
if (range.start !== start) { range.start = start }
if (range.end !== end) { range.end = end }
"""

# Double-click reset of the map view. Only a double-click ON the map counts:
# a double-click on a colorbar resets that column's colour range instead
# (see _COLORBAR_DRAG_JS), and must not also throw away the zoom.
#
# Bokeh's ResetTool only fires from its toolbar button,
# and the toolbar is suppressed (see layout()), so double-click has to be
# wired up by hand. Resets EVERY panel, not just the clicked one - see
# _wire_dblclick_reset.
#
# This calls each plot VIEW's reset() - what ResetTool itself does - rather
# than setting range start/end directly. Setting the ranges does move the
# axes, but doesn't emit Bokeh's RangesUpdate event, which is the only thing
# HoloViews' RangeXY stream listens to. rasterize() then never re-aggregates,
# and the "reset" view shows just the patch rasterized for the old zoom.
#
# The figures to reset are found in the document by `token` (see
# _wire_dblclick_reset) rather than passed in as args: a callback whose args
# hold the very figures it's attached to is a circular reference Bokeh
# refuses to serialize.
_RESET_JS = """
const origin_view = Bokeh.index.find_one(cb_obj.origin)
if (origin_view == null || !origin_view.frame.bbox.contains(cb_obj.sx, cb_obj.sy)) { return }
for (const m of cb_obj.origin.document.all_models) {
  const cbs = (m.js_event_callbacks || {}).doubletap || []
  if (!cbs.some(c => c.args && c.args.token === token)) { continue }
  const view = Bokeh.index.find_one(m)
  if (view != null) { view.reset() }
}
"""

# Colorbar tick layout - see _colorbar_ticks. The colorbar's ends are
# always ticked, so the exact min/max are readable, plus TICK_BAND "nice"
# interior ticks. Interior ticks closer than END_TICK_CLEARANCE (as a
# fraction of the range) to an end are dropped so their labels don't
# overlap the end labels.
#
# The step is whichever candidate lands the interior count inside
# TICK_BAND, roundest first (TICK_MANTISSAS is in order of preference).
# A plain 1/2/5 ladder can't do that: as a drag stretches the range it
# swings between 2 and 6 interior ticks, jumping by up to 3 in one mouse
# move where the step flips. The in-between steps (2.5, 4, 3, 1.5, 6, 8)
# are what hold the count to the band. Simulated over 20k random ranges
# the count stays within the band in all but ~0.6% (one tick over), and a
# continuous drag never changes it by more than one tick at a time.
TICK_BAND = (4, 5)
TICK_MANTISSAS = (1, 2, 5, 2.5, 4, 3, 1.5, 6, 8)
END_TICK_CLEARANCE = 0.07

# JS twin of _colorbar_ticks, used to re-tick while a drag is in progress.
# Keep the two in step.
_TICKS_JS = """
function endTicks(lo, hi) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) { return [] }
  const [b0, b1] = %s
  const mantissas = %s
  const clear = %r * (hi - lo)
  const e = Math.floor(Math.log10(hi - lo))
  let best = null
  for (let k = e - 2; k <= e; k++) {
    mantissas.forEach((m, rank) => {
      const step = m * Math.pow(10, k)
      const inner = []
      for (let j = Math.ceil(lo / step); j * step <= hi; j++) {
        const t = j * step
        if (t - lo > clear && hi - t > clear) { inner.push(t) }
      }
      const n = inner.length
      const d = n < b0 ? b0 - n : n > b1 ? n - b1 : 0
      if (best == null || d < best.d || (d === best.d && (rank < best.rank ||
          (rank === best.rank && k > best.k)))) {
        best = {d, rank, k, inner}
      }
    })
  }
  return [lo, ...best.inner, hi]
}
""" % (list(TICK_BAND), list(TICK_MANTISSAS), END_TICK_CLEARANCE)

# Tick labels: interior ticks get as many decimals as their spacing needs.
# The end ticks (arbitrary values like 222.339996) get one more, and at
# least two significant digits so a small end like -0.000074 doesn't round
# to a misleading -0.0001 - capped at three extra so a near-zero end can't
# grow a long string of decimals. Bokeh's default would instead format
# every label to the precision of the least-round one.
_TICK_FORMAT_JS = """
const n = ticks.length
const step = n >= 4 ? Math.abs(ticks[2] - ticks[1]) : Math.abs(ticks[n - 1] - ticks[0])
const end = index === 0 || index === n - 1
if (!(step > 0)) { return tick.toPrecision(4) }
if (Math.abs(tick) < step * 1e-9) { return "0" }
if (step < 1e-4 || Math.abs(tick) >= 1e6) { return tick.toExponential(end ? 2 : 1) }
let dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9))
// A 2.5 or 1.5 step needs a decimal more than its magnitude suggests.
while (dec < 12 && Math.abs(step * 10 ** dec - Math.round(step * 10 ** dec)) > 1e-6) { dec++ }
if (end) {
  const sig = tick === 0 ? 0 : 1 - Math.floor(Math.log10(Math.abs(tick)))
  dec = Math.min(Math.max(dec + 1, sig), dec + 3)
}
return tick.toFixed(dec)
"""

# Colorbar drag-to-rescale and cursor feedback - see _wire_colorbar_drag.
#
# Grab a value on the bar and drag it: grabbed in the upper half, the
# bottom end stays put and the top stretches so the grabbed value stays
# under the cursor (drag 250 up to the top and the range becomes lo..250);
# lower half is the mirror image. Values outside the new range saturate at
# the end colours, which is Bokeh's LinearColorMapper default.
#
# Runs entirely in the browser while dragging, updating the colour mapper
# and ticks of every colorbar in the same column (tagged with `token`) so
# the column rescales together without a server round trip per mouse move.
# On release it writes the final range to `commit`, a ColumnDataSource used
# purely as a browser -> Python channel - see _on_colorbar_commit. Its `n`
# column is a nonce, so repeating an identical message still registers as
# a change. Double-clicking a colorbar sends a "reset" the same way. (A custom
# Bokeh DataModel would read better, but BokehJS can't resolve one that
# first appears in a document after the page has loaded.)
#
# f is clamped away from 0 so dragging the grabbed value right down to the
# fixed end can't divide by zero or flip the range. The release commits the
# range from the last drag move, not from the release event's own position:
# BokehJS can emit a stray panend carrying a bogus position (seen right
# after a synthetic double-click), which would otherwise commit a wrong
# range.
#
# Cursor: ns-resize over and while dragging the colorbar, grab over the map,
# grabbing while panning it, and zoom-in/zoom-out briefly on a wheel zoom.
# Bokeh itself only shows a cursor for the active *move* tool, which here is
# none. It resets the cursor on every mouse move before emitting the
# figure's MouseMove event, so setting it from that event sticks.
_COLORBAR_DRAG_JS = _TICKS_JS + """
const fig = cb_obj.origin
const pv = Bokeh.index.find_one(fig)
if (pv == null) { return }
const st = pv.__cbar_drag || (pv.__cbar_drag = {active: false, panning: false})
const ev = cb_obj.event_name
const sx = cb_obj.sx
const sy = cb_obj.sy
const setCursor = c => pv.canvas_view.ui_event_bus.set_cursor(c)

function barAt(sx, sy) {
  const panels = [...fig.left, ...fig.right, ...fig.above, ...fig.below, ...fig.center]
  const cbar = panels.find(m => (m.tags || []).includes(token))
  if (cbar == null || cbar.orientation === "horizontal") { return null }
  const cbv = Bokeh.index.find_one(cbar)
  if (cbv == null || cbv._inner_layout == null) { return null }
  const lo = cbar.color_mapper.low
  const hi = cbar.color_mapper.high
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) { return null }
  const bar = cbv._inner_layout.center_panel.bbox
  if (!cbv._inner_layout.bbox.contains(sx, sy)) { return null }
  if (sy < bar.top - 6 || sy > bar.bottom + 6) { return null }
  return {lo, hi, bar}
}
function hoverCursor() {
  if (barAt(sx, sy) != null) { return "ns-resize" }
  if (pv.frame.bbox.contains(sx, sy)) { return "grab" }
  return null
}

if (ev === "mousemove") {
  setCursor(st.active ? "ns-resize" : st.panning ? "grabbing" : hoverCursor())
  return
}
if (ev === "doubletap") {
  if (barAt(sx, sy) != null) {
    commit.data = {kind: [kind], action: ["reset"], low: [0], high: [0], n: [Date.now()]}
  }
  return
}
if (ev === "wheel") {
  if (!pv.frame.bbox.contains(sx, sy)) { return }
  setCursor(cb_obj.delta > 0 ? "zoom-in" : "zoom-out")
  clearTimeout(st.wheel_timer)
  st.wheel_timer = setTimeout(() => setCursor(hoverCursor()), 300)
  return
}
if (ev === "panstart") {
  st.active = false
  st.panning = false
  const hit = barAt(sx, sy)
  if (hit == null) {
    if (pv.frame.bbox.contains(sx, sy)) {
      st.panning = true
      setCursor("grabbing")
    }
    return
  }
  const {lo, hi, bar} = hit
  const f0 = Math.min(Math.max((bar.bottom - sy) / bar.height, 0), 1)
  const peers = []
  for (const m of fig.document.all_models) {
    if ((m.tags || []).includes(token) && m.color_mapper != null) { peers.push(m) }
  }
  Object.assign(st, {active: true, lo, hi, bottom: bar.bottom,
    height: bar.height, v: lo + f0 * (hi - lo), upper: f0 >= 0.5, peers,
    last: [lo, hi]})
  setCursor("ns-resize")
  return
}
if (st.panning) {
  if (ev === "panend") {
    st.panning = false
    setCursor(hoverCursor())
  } else {
    setCursor("grabbing")
  }
  return
}
if (!st.active) { return }

if (ev === "panend") {
  st.active = false
  commit.data = {kind: [kind], action: ["set"], low: [st.last[0]],
    high: [st.last[1]], n: [Date.now()]}
  setCursor(hoverCursor())
  return
}

const f = (st.bottom - sy) / st.height
let lo = st.lo
let hi = st.hi
if (st.upper) {
  hi = st.lo + (st.v - st.lo) / Math.min(Math.max(f, 0.05), 20)
} else {
  lo = st.hi - (st.hi - st.v) / Math.min(Math.max(1 - f, 0.05), 20)
}
const ticks = endTicks(lo, hi)
for (const cb of st.peers) {
  cb.color_mapper.setv({low: lo, high: hi})
  if (cb.ticker != null && cb.ticker.constructor.__name__ === "FixedTicker") {
    cb.ticker.ticks = ticks
  }
}
st.last = [lo, hi]
setCursor("ns-resize")
"""


def _colorbar_ticks(lo, hi):
    """Tick positions for a colorbar spanning lo..hi: both ends, plus the
    interior ticks chosen as described at TICK_BAND. Python twin of
    _TICKS_JS - keep the two in step."""
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return []
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return []
    b0, b1 = TICK_BAND
    clear = END_TICK_CLEARANCE * (hi - lo)
    e = int(np.floor(np.log10(hi - lo)))
    best = None
    for k in range(e - 2, e + 1):
        for rank, m in enumerate(TICK_MANTISSAS):
            step = m * 10.0 ** k
            inner = [float(j * step)
                     for j in range(int(np.ceil(lo / step)),
                                    int(np.floor(hi / step)) + 1)]
            inner = [t for t in inner if t - lo > clear and hi - t > clear]
            n = len(inner)
            d = b0 - n if n < b0 else n - b1 if n > b1 else 0
            key = (d, rank, -k)
            if best is None or key < best[0]:
                best = (key, inner)
    return [lo, *best[1], hi]


# Value-dimension name for placeholder panels, so _sync_colorbar_title can
# tell them apart from real fields - a real field can't be told apart by
# HoloViews' default "z", because z (geopotential) is a real variable.
PLACEHOLDER_VDIM = "__placeholder__"

# "No explicit colour limits" - HoloViews reads (None, None) as autoscale
# from the data.
#
# This is deliberately NOT (nan, nan). A NaN clim produces a colour mapper
# with no usable range: every value maps to the low end of the colormap (a
# flat blue rectangle for coolwarm) and the colorbar is suppressed entirely,
# because there is no range to label. With None the same situation degrades
# to a correctly autoscaled panel instead.
CLIM_UNSET = (None, None)


class PlotGridState(param.Parameterized):
    """Shared state for the grid. Bind the existing sidebar widgets to this
    rather than to the plots directly (see app_layout.link_controls)."""

    # Unbounded deliberately: load_e2s_field clamps t to the available range,
    # so a stale or out-of-range index degrades to the last frame rather than
    # raising. Declaring bounds here would instead raise ValueError from
    # inside a param watcher during a suite switch, when the slider's new
    # value can arrive before set_time_bounds has run.
    time_index = param.Integer(default=0)

    variable = param.String(default="q")
    level = param.Integer(default=500, allow_None=True)

    # Either a colormap name or a matplotlib Colormap object - cmocean's
    # colormaps are not registered globally, so they arrive as objects.
    cmap = param.Parameter(default=DEFAULT_CMAP)
    cmap_min = param.Number(default=0.0)
    cmap_max = param.Number(default=0.0)

    # Colour limits, recomputed on variable/level change rather than per
    # frame. Per-frame autoscaling makes an animation unreadable, because
    # the scale slides underneath the data.
    field_clim = param.Tuple(default=CLIM_UNSET, length=2)
    diff_clim = param.Tuple(default=CLIM_UNSET, length=2)

    # model name -> other model name (or None for "no difference selected")
    diff_pairs = param.Dict(default={})

    # model name -> "ready" | "computing" | "error: ..."
    diff_status = param.Dict(default={})

    # Valid time of the currently displayed step, published by whichever
    # field panel is nominated first.
    header_text = param.String(default="")

    # A visualization.boundaries.BOUNDARY_OPTIONS label. In neither stream
    # list below: the lines live in their own data sources (see
    # PlotGrid._draw_boundaries), so changing them re-renders nothing.
    boundaries = param.String(default=BOUNDARIES_DEFAULT)

    # Cursor readout rows, as (label, value) pairs in two columns:
    # (left_pairs, right_pairs). Empty until the cursor first enters a
    # panel. Stored as data rather than rendered HTML so that a time-slider
    # tick can re-render the timestamp beside the existing values without
    # waiting for the next mouse move.
    readout_rows = param.Tuple(default=((), ()), length=2)


# Which parameters each half of the grid actually depends on.
#
# Two streams rather than one: nothing in _field_cb reads diff_pairs or
# diff_status, so putting them on a shared stream would make a background
# diff landing, or a selector change, force six 721x1440 reloads of field
# data that hasn't changed.
#
# cmap, cmap_min/max, the clim tuples and readout_rows appear in NEITHER
# list - those are restyles or pure display, and must not trigger a re-read.
_FIELD_PARAMS = ["time_index", "variable", "level"]
_DIFF_PARAMS = ["time_index", "variable", "level", "diff_pairs", "diff_status"]


def _schedule(fn):
    """Run fn on the Bokeh document's event loop.

    Param values touched from a worker thread must not be mutated directly:
    Panel needs the document lock to push the resulting change to the
    browser. pn.state.execute handles that when a server session exists,
    and falls through to a direct call in scripts and tests.
    """
    try:
        if pn.state.curdoc is not None:
            pn.state.execute(fn)
            return
    except Exception:
        pass
    fn()


def _fmt_lat(lat):
    """45.62 -> "45.62°N", -45.62 -> "45.62°S"."""
    return f"{abs(lat):.2f}\u00b0{'N' if lat >= 0 else 'S'}"


def _fmt_lon(lon):
    """Longitude as °E/°W, whether the grid runs 0..360 or -180..180."""
    lon = (lon + 180) % 360 - 180
    return f"{abs(lon):.2f}\u00b0{'E' if lon >= 0 else 'W'}"


def _fmt(value):
    """Format a field value for the cursor readout.

    %.4g would switch to scientific notation below 1e-4, which for specific
    humidity differences is most values - and those strings are wide enough
    to wrap the readout column, which grows the whole grid and pushes every
    row below it down. Fixed-point where possible keeps the width
    predictable and the decimal points aligned.
    """
    if value is None or not np.isfinite(value):
        return "\u2014"
    a = abs(value)
    if a >= 1000 or (a > 0 and a < 1e-6):
        return f"{value:.3e}"      # genuinely needs an exponent
    if a >= 1:
        return f"{value:.3f}"
    return f"{value:.7f}"


class PlotGrid(param.Parameterized):
    """Builds and owns the linked plot grid."""

    def __init__(self, models, model_dirs, state=None):
        """
        models         : list[str] - model names, in display order
        model_dirs     : dict[str, Path] - model name -> directory of .nc files
        state          : PlotGridState, or None to create one
        """
        super().__init__()
        self.models = list(models)
        self.model_dirs = dict(model_dirs)
        self.state = state if state is not None else PlotGridState()

        # Forecast length, set by set_time_bounds. Only frame_spec reads it;
        # the panels themselves rely on load_e2s_field's clamping.
        self.n_steps = 1

        # (kind, model) -> (DataArray, FieldMeta) for whatever each panel is
        # currently displaying. The readout samples these rather than
        # re-reading from disk, which is what makes a per-mouse-move lookup
        # affordable: it is an in-memory .sel on a 2D array.
        self._fields = {}
        self._fields_lock = threading.Lock()

        self._last_pointer = 0.0
        self._pointer_streams = []

        # Extent of the most recently loaded field, so placeholder panels can
        # match it. A placeholder with a different extent would drag every
        # panel to its range once the axes are linked - which is why this is
        # seeded from a real field here rather than left on the constant: the
        # first render is exactly when placeholder difference panels sit next
        # to real field panels.
        self._last_extent = GLOBAL_EXTENT
        try:
            da, meta = load_e2s_field(
                self.model_dirs[self.models[0]],
                self.state.variable, self.state.level, 0)
            lon, lat = da[meta.lon_dim].values, da[meta.lat_dim].values
            self._last_extent = (float(lon.min()), float(lat.min()),
                                 float(lon.max()), float(lat.max()))
        except Exception:
            pass   # constant fallback is fine; extent corrects on first render

        # Guards against spawning duplicate background jobs for one pair.
        self._pending = set()
        self._pending_lock = threading.Lock()

        # Set once this grid has been replaced, so any callback still in
        # flight when teardown() runs bails out instead of loading a field
        # from a suite that is no longer on screen.
        self._torn_down = False

        self._hv_pane = None

        # id(Range1d) -> CustomJS enforcing that range's zoom/pan limits -
        # see _apply_zoom_clamp. Cleared in teardown() so a torn-down grid
        # doesn't keep these ranges (and the fields captured in their
        # CustomJS args) alive.
        self._zoom_customjs = {}

        # Double-click reset: one CustomJS shared by every panel, and the
        # ids of the figures it's already attached to - see
        # _wire_dblclick_reset. The token is unique per grid, so a stale
        # grid's figures can't be reset along with this one's.
        self._reset_customjs = CustomJS(
            args=dict(token=f"plotgrid-{id(self)}"), code=_RESET_JS)
        self._reset_figs = set()

        # Colorbar drag: one CustomJS per column kind ("field"/"diff"),
        # each with its own token so dragging one column's colorbar never
        # rescales the other. Both report back through the same commit
        # model - see _wire_colorbar_drag.
        self._cbar_commit = ColumnDataSource(
            data=dict(kind=[], action=[], low=[], high=[], n=[]))
        self._cbar_commit.on_change("data", self._on_colorbar_commit)
        self._cbar_drag_customjs = {
            kind: CustomJS(
                args=dict(token=self._cbar_token(kind), kind=kind,
                          commit=self._cbar_commit),
                code=_COLORBAR_DRAG_JS)
            for kind in ("field", "diff")
        }
        self._cbar_drag_figs = set()

        # kind -> (key, clim) for the last automatically computed range, so
        # a colorbar double-click can restore it without re-reading data -
        # see _reset_clim.
        self._auto_clims = {}

        self._field_stream = hv.streams.Params(self.state, _FIELD_PARAMS)
        self._diff_stream = hv.streams.Params(self.state, _DIFF_PARAMS)
        self._layout = None

        # Set by diff_selectors; kept so teardown can unwatch. The state
        # outlives every grid built against it, so a leaked watcher would
        # keep firing on a dead grid and mutate widgets no longer on screen.
        self._status_watcher = None

        # Recompute colour limits whenever the user picks a different
        # variable or level. Without this, field_clim/diff_clim are only
        # ever set at grid construction (app_layout's refresh_clims_async)
        # and when a difference finishes computing, so switching variables
        # left the scale stuck at whatever the previous variable needed.
        self._clim_watcher = self.state.param.watch(
            lambda event: self.refresh_clims_async(), ["variable", "level"])

        # Boundary lines: one data source per longitude convention (False
        # for -180..180 maps, True for 0..360), each shared by every panel
        # drawn in that convention. _boundary_used says which conventions
        # have panels, so only those get the line data sent to the browser.
        self._boundary_sources = {
            lon360: ColumnDataSource(data=dict(x=np.empty(0, np.float32),
                                               y=np.empty(0, np.float32)))
            for lon360 in (False, True)
        }
        self._boundary_used = set()
        self._boundary_figs = set()
        self._boundary_watcher = self.state.param.watch(
            lambda event: self._update_boundaries(), "boundaries")

    # -- lifecycle ------------------------------------------------------

    def teardown(self):
        """Detach this grid from the shared state.

        A stream holds a subscription to the PlotGridState for as long as it
        exists, and the state deliberately outlives every grid built against
        it (so the sidebar never needs rebinding). Without this, a replaced
        grid keeps servicing every parameter change off-screen - reloading
        fields from the previous suite's directories on each slider tick, and
        racing the live grid to write the readout.
        """
        self._torn_down = True

        if self._status_watcher is not None:
            try:
                self.state.param.unwatch(self._status_watcher)
            except Exception:
                pass
            self._status_watcher = None

        if self._clim_watcher is not None:
            try:
                self.state.param.unwatch(self._clim_watcher)
            except Exception:
                pass
            self._clim_watcher = None

        if self._boundary_watcher is not None:
            try:
                self.state.param.unwatch(self._boundary_watcher)
            except Exception:
                pass
            self._boundary_watcher = None

        for stream in (self._field_stream, self._diff_stream,
                       *self._pointer_streams):
            try:
                stream.clear()          # drop subscribers
            except Exception:
                pass
            try:
                stream.source = None    # drop the reference
            except Exception:
                pass
        self._pointer_streams = []
        with self._fields_lock:
            self._fields.clear()
        self._zoom_customjs.clear()
        self._reset_figs.clear()
        self._cbar_drag_figs.clear()
        self._boundary_figs.clear()
        self._layout = None
        self._hv_pane = None

        # A pair still computing when this grid is replaced never reports
        # back (work() bails out once _torn_down is set), so its "computing"
        # would otherwise sit on the shared state forever - and keep the
        # app's Visualization-tab busy overlay up with it. Last, so the
        # streams cleared above don't re-render this grid in response.
        self.state.diff_status = {}

    # -- element construction ------------------------------------------

    def _sizing_opts(self):
        """Sizing opts shared by real and placeholder panels.

        Fixed frame, not responsive=True: the two are mutually exclusive in
        Bokeh. See PANEL_FRAME_WIDTH for why fixed won.
        """
        return dict(
            frame_width=PANEL_FRAME_WIDTH,
            frame_height=PANEL_FRAME_HEIGHT,
        )

    def _clamp_zoom_pan(self, plot, element):
        """Bokeh hook (see _panel_opts) that caps how far a panel can be
        zoomed/panned, on the actual rendered figure rather than through
        HoloViews' xlim/ylim - those only set the initial view, not a hard
        limit, and the underlying Range1d is what wheel_zoom/pan act on.

        Split across two mechanisms rather than one - see _apply_zoom_clamp
        and _PAN_CLAMP_JS for why.

        Deliberately reads bounds from self._last_extent, NOT from
        `element.range(...)`. This hook re-fires on every zoom/pan, because
        rasterize() re-aggregates dynamically over whatever the current
        viewport is - so the `element` handed to a hook after a zoom/pan
        describes that just-zoomed window, not the full field. Using its
        range here fed each zoom/pan back in as the new "full extent",
        freezing max_interval at whatever size you'd just zoomed to and
        collapsing the pan bounds to that same window - the exact "can't
        zoom out, pan snaps back" loop this replaced. self._last_extent
        comes from the source DataArray's own coordinate array (see
        _element), which doesn't change with the current view.

        Bounds come from the current field's real lon/lat range rather than
        an arbitrary global one - a model writing lon on 0..360 gets bounds
        on 0..360, not -180..180. min_span caps zooming in below a window
        that would leave only a handful of rasterized blocks on screen.
        """
        lon0, lat0, lon1, lat1 = self._last_extent
        fig = plot.state
        self._apply_zoom_clamp(fig.x_range, lon0, lon1, MIN_LON_SPAN)
        self._apply_zoom_clamp(fig.y_range, lat0, lat1, MIN_LAT_SPAN)

    def _apply_zoom_clamp(self, rng, lo, hi, min_span):
        """Cap `rng` at [lo, hi] and no tighter than min_span, via two
        independent mechanisms:

        - min_interval/max_interval, Bokeh's own native span limit - NOT
          Range1d.bounds. Every report of WheelZoomTool getting stuck and
          unable to zoom back out (bokeh/bokeh#6950, #8118, #10440, #11294)
          is specifically about `.bounds`; min/max_interval alone is the
          documented, working mechanism for this and isn't implicated.
        - a CustomJS pan-only clamp (_PAN_CLAMP_JS) for the case `.bounds`
          would otherwise cover: a drag pushing the (already span-limited)
          window past lo/hi. Registering a new callback on every hook call
          (this runs on every re-render) would pile up duplicate listeners
          on the same range, so an existing callback's `args` is mutated in
          place instead of adding another.
        """
        rng.min_interval = min(min_span, hi - lo)
        rng.max_interval = hi - lo

        cb = self._zoom_customjs.get(id(rng))
        if cb is None:
            cb = CustomJS(args=dict(range=rng, lo=lo, hi=hi), code=_PAN_CLAMP_JS)
            rng.js_on_change("start", cb)
            rng.js_on_change("end", cb)
            self._zoom_customjs[id(rng)] = cb
        else:
            cb.args = dict(range=rng, lo=lo, hi=hi)

    def _disable_axis_zoom(self, plot, element):
        """Bokeh hook (see _panel_opts): scrolling over an axis zooms the
        whole map, not just that one axis.

        WheelZoomTool's zoom_on_axis (on by default) makes a scroll over the
        lon or lat axis stretch only that dimension, distorting the map's
        aspect. Set on the rendered tool rather than by passing a
        WheelZoomTool instance in `tools`, which would hand one Bokeh model
        to every panel's figure.
        """
        for tool in plot.state.tools:
            if isinstance(tool, WheelZoomTool) and tool.zoom_on_axis:
                tool.zoom_on_axis = False

    def _wire_dblclick_reset(self, plot, element):
        """Bokeh hook (see _panel_opts): double-click resets every panel.
        See _RESET_JS.

        Every panel, not just the clicked one: shared_axes only links
        pairwise (see the module docstring), so resetting one panel could
        leave the other pair zoomed in. The hook runs on every re-render,
        so the shared callback is attached once per figure, not per call.
        """
        fig = plot.state
        if id(fig) not in self._reset_figs:
            fig.js_on_event("doubletap", self._reset_customjs)
            self._reset_figs.add(id(fig))

    @staticmethod
    def _tag_autosize(plot, element):
        """Bokeh hook (see _panel_opts): mark the figure for
        PlotGridAutoSize (app_layout.py) to widen with the window."""
        fig = plot.state
        if AUTOSIZE_TAG not in fig.tags:
            fig.tags = [*fig.tags, AUTOSIZE_TAG]

    def _draw_boundaries(self, plot, element):
        """Bokeh hook (see _panel_opts): draw the selected Natural Earth
        boundaries over the map.

        A plain Bokeh line glyph added to the figure, rather than a
        HoloViews Path overlaid on the field: the lines then sit outside
        HoloViews' rendering entirely, so changing them never re-renders or
        re-rasterizes a panel, and they can't disturb the ranges, zoom
        clamp or colorbar wiring the other hooks set up. The data source is
        shared, so every panel updates at once. Attached once per figure;
        added last, so it draws on top of the field.
        """
        fig = plot.state
        if id(fig) in self._boundary_figs:
            return
        try:
            lon360 = float(element.range(0)[1]) > 180
        except Exception:
            lon360 = self._last_extent[2] > 180
        source = self._boundary_sources[lon360]
        fig.line("x", "y", source=source, line_color=BOUNDARY_HALO_COLOR,
                 line_width=BOUNDARY_HALO_WIDTH,
                 line_alpha=BOUNDARY_HALO_ALPHA)
        fig.line("x", "y", source=source, line_color=BOUNDARY_COLOR,
                 line_width=BOUNDARY_WIDTH)
        self._boundary_figs.add(id(fig))
        if lon360 not in self._boundary_used:
            self._boundary_used.add(lon360)
            self._update_boundaries(only=lon360)

    def _update_boundaries(self, only=None):
        """Load the selected boundaries into the data sources in use."""
        for lon360 in self._boundary_used:
            if only is not None and lon360 != only:
                continue
            x, y = boundary_lines(self.state.boundaries, lon360)
            self._boundary_sources[lon360].data = dict(x=x, y=y)

    def _cbar_token(self, kind):
        return f"plotgrid-{id(self)}-cbar-{kind}"

    def _wire_colorbar_drag(self, kind, plot, element):
        """Bokeh hook (see _panel_opts): drag a value on the colorbar to
        rescale that column's colour range. See _COLORBAR_DRAG_JS.

        The colorbar is tagged with its column's token, which is how the
        JS finds the colorbar under the cursor and every peer that has to
        rescale with it. Re-tagged on every call in case HoloViews rebuilt
        the colorbar; the drag callback itself is attached once per figure.
        """
        colorbar = plot.handles.get("colorbar")
        if colorbar is None:
            return
        token = self._cbar_token(kind)
        if token not in colorbar.tags:
            colorbar.tags = [*colorbar.tags, token]
        fig = plot.state
        if id(fig) not in self._cbar_drag_figs:
            for event in ("panstart", "pan", "panend", "mousemove", "wheel",
                          "doubletap"):
                fig.js_on_event(event, self._cbar_drag_customjs[kind])
            self._cbar_drag_figs.add(id(fig))

    def _tick_colorbar_ends(self, plot, element):
        """Bokeh hook (see _panel_opts): tick the colorbar's exact min and
        max as well as the usual round values. See _colorbar_ticks.

        Recomputed on every render, since the range changes with the
        variable, the sidebar or a colorbar drag; during a drag the JS
        re-ticks in step (_COLORBAR_DRAG_JS). The ticker and formatter are
        swapped in once per colorbar and only their ticks updated after.
        """
        colorbar = plot.handles.get("colorbar")
        if colorbar is None:
            return
        cm = colorbar.color_mapper
        ticks = _colorbar_ticks(cm.low, cm.high)
        if not isinstance(colorbar.ticker, FixedTicker):
            colorbar.ticker = FixedTicker(ticks=ticks, minor_ticks=[])
            colorbar.formatter = CustomJSTickFormatter(code=_TICK_FORMAT_JS)
        elif list(colorbar.ticker.ticks) != ticks:
            colorbar.ticker.ticks = ticks

    def _on_colorbar_commit(self, attr, old, new):
        """A colorbar drag finished, or a colorbar was double-clicked.

        A drag ("set") makes its range the column's clim. Writing the clim
        param, rather than leaving the browser-side mapper change in place,
        is what makes the range survive re-renders and time-slider ticks,
        and lets app_layout reflect it into the sidebar. A double-click
        ("reset") puts back the automatic range - see _reset_clim. The next
        variable/level change recomputes it either way.
        """
        if self._torn_down:
            return
        try:
            kind, action = new["kind"][0], new["action"][0]
            lo, hi = float(new["low"][0]), float(new["high"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            return
        if action == "reset":
            self._reset_clim(kind)
            return
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return
        name = "diff_clim" if kind == "diff" else "field_clim"
        setattr(self.state, name, (lo, hi))

    def _field_clim_key(self):
        return (self.state.variable, self.state.level)

    def _diff_clim_key(self):
        return (self.state.variable, self.state.level,
                tuple(sorted(self.state.diff_pairs.items())))

    def _reset_clim(self, kind):
        """Put a column's automatic colour range back after a colorbar drag.

        Uses the range the last refresh computed when it's still for the
        variable/level (and, for differences, pairs) on screen, so the reset
        is immediate; otherwise recomputes it in the background as a
        variable change would. An explicit sidebar Min/Max still wins for
        the field column, as it does in _refresh_field_clim.
        """
        if kind == "diff":
            cached = self._auto_clims.get("diff")
            if cached is not None and cached[0] == self._diff_clim_key():
                self.state.diff_clim = cached[1]
            else:
                self.refresh_diff_clim_async()
            return
        explicit = self.state.cmap_min != 0.0 or self.state.cmap_max != 0.0
        cached = self._auto_clims.get("field")
        if not explicit and cached is not None and cached[0] == self._field_clim_key():
            self.state.field_clim = cached[1]
        else:
            self.refresh_field_clim_async()

    def _sync_colorbar_title(self, plot, element):
        """Bokeh hook (see _panel_opts): keep the colorbar label in step with
        the variable on screen.

        HoloViews titles the colorbar once, from the FIRST element a
        DynamicMap yields, and only swaps the data after that (the same
        first-element rule _panel_opts describes). Without this, switching
        variable left the colorbar reading the first variable's name and
        units. Placeholders get no label rather than a meaningless one.
        """
        colorbar = plot.handles.get("colorbar")
        if colorbar is None or not element.vdims:
            return
        vdim = element.vdims[0]
        title = "" if vdim.name == PLACEHOLDER_VDIM else vdim.pprint_label
        if colorbar.title != title:
            colorbar.title = title

    def _panel_opts(self, title, cmap, kind):
        """Every option that defines a panel's STRUCTURE.

        _element and _placeholder must both go through here, and the result
        must not differ between them beyond title and colormap. HoloViews
        builds a subplot's structure from the FIRST element its DynamicMap
        yields and only swaps the data thereafter - it does not add a
        colorbar that was not there, or attach tools that were not there.

        That is precisely what broke the difference panels: on first render
        no pair is selected, so their DynamicMaps yielded a placeholder, and
        the placeholder set colorbar=False and tools=[]. The panels then
        stayed colorbar-less and un-navigable even once real difference data
        arrived, while the field panels - which yield a real element first -
        were fine.

        default_tools=[] is load-bearing too. HoloViews PREPENDS its own set
        (pan, wheel_zoom, box_zoom, save, reset, help) and `tools` only ADDS
        to it, so without this the plot gets a second wheel_zoom alongside
        ours and box_zoom reappears despite never being asked for.

        Note there is no `toolbar` key: toolbar is a Layout-level option in
        HoloViews. Setting it on an element does not suppress the strip and
        does interfere with tool activation. See layout().

        `kind` ("field" or "diff") says which column the panel belongs to,
        so a colorbar drag rescales only that column - see
        _wire_colorbar_drag.
        """
        return dict(
            title=title,
            cmap=cmap,
            colorbar=True,
            colorbar_opts={"width": COLORBAR_WIDTH},
            default_tools=[],
            tools=["pan", "wheel_zoom", "reset"],
            active_tools=["pan", "wheel_zoom"],
            # framewise=False is what makes zoom survive a time-slider tick:
            # the axis ranges are not recomputed when the data changes.
            framewise=False,
            shared_axes=True,
            xlabel="longitude",
            ylabel="latitude",
            hooks=[self._tag_autosize,
                   self._clamp_zoom_pan, self._disable_axis_zoom,
                   self._wire_dblclick_reset,
                   self._sync_colorbar_title, self._tick_colorbar_ends,
                   partial(self._wire_colorbar_drag, kind),
                   self._draw_boundaries],
            **self._sizing_opts(),
        )

    def _element(self, da, meta, title, cmap=DEFAULT_CMAP, kind="field"):
        """Wrap a loaded field as the appropriate HoloViews element.

        The field panels' colormap comes from the .apply.opts() chain in
        layout() as a param reference, so the cmap passed here is only a
        structural placeholder for them. The difference panels pass
        DIFF_CMAP explicitly, because a static value in that chain alongside
        param references makes HoloViews rebuild the branch differently.
        """
        lon = da[meta.lon_dim].values
        lat = da[meta.lat_dim].values
        self._last_extent = (
            float(lon.min()), float(lat.min()),
            float(lon.max()), float(lat.max()),
        )

        # hv.Image assumes an evenly spaced grid and will silently misplace
        # data on, say, a reduced Gaussian latitude axis. QuadMesh handles
        # irregular spacing correctly, at some rendering cost.
        #
        # kdims come from meta, which earth2StudioPlot has already
        # canonicalized to latitude/longitude regardless of what a given
        # model called them on disk.
        cls = hv.Image if meta.regular_grid else hv.QuadMesh

        return cls(da, kdims=[meta.lon_dim, meta.lat_dim]).opts(
            **self._panel_opts(title, cmap, kind))

    def _placeholder(self, title, computing=False, kind="field"):
        """Blank panel that preserves grid geometry, axis ranges, and plot
        structure - see _panel_opts on why the third of those matters.

        The kdims are NOT optional and NOT cosmetic. shared_axes links
        panels by DIMENSION NAME, and a placeholder built from a bare array
        would take HoloViews' default x/y names - landing it in a different
        group from the real panels on longitude/latitude. That is exactly
        what split the grid into two pairs (fields together, placeholders
        together) and cost a long hunt to find. earth2StudioPlot
        canonicalizes every real field to these same names, so reusing its
        constants keeps the two in step.

        NaN rather than zeros: a placeholder carries a real colormap (it has
        to, for the structure to match), and zeros would render as a solid
        block of that colormap's low end. NaN renders transparent, which
        reads as empty.

        `computing` gives the in-progress state its own appearance. Without
        it a panel waiting on a multi-minute difference computation looks
        identical to one with nothing selected, which reads as "broken"
        rather than "working".
        """
        left, bottom, right, top = self._last_extent
        cmap = COMPUTING_CMAP if computing else PLACEHOLDER_CMAP
        opts = self._panel_opts(title, cmap, kind)
        if computing:
            opts["title"] = f"\u27f3  {title}"
            opts["fontsize"] = {"title": "13pt"}
        return hv.Image(
            np.full((2, 2), np.nan), bounds=(left, bottom, right, top),
            kdims=[CANON_LON, CANON_LAT], vdims=[PLACEHOLDER_VDIM],
        ).opts(**opts)

    # -- callbacks ------------------------------------------------------

    def _field_cb(self, model, is_first, **_):
        if self._torn_down:
            return self._placeholder(model)

        try:
            da, meta = load_e2s_field(
                self.model_dirs[model],
                self.state.variable,
                self.state.level,
                self.state.time_index,
            )
        except Exception as exc:
            # An exception escaping a DynamicMap callback can leave the pane
            # permanently broken, so failures are rendered as a titled blank
            # panel instead. Full traceback goes to the server log.
            traceback.print_exc()
            with self._fields_lock:
                self._fields.pop(("field", model), None)
            if isinstance(exc, FieldReadError):
                # Already retried on a reopened file. Any change of time,
                # variable or level re-runs this callback, i.e. tries again.
                return self._placeholder(
                    f"{model} - {exc.short} - change time to retry")
            return self._placeholder(f"{model} - {type(exc).__name__}: {exc}")

        with self._fields_lock:
            self._fields[("field", model)] = (da, meta)

        if is_first:
            # One panel is nominated to publish the shared valid-time
            # header, avoiding an extra read purely to populate it.
            label = meta.time_label()
            if label != self.state.header_text:
                _schedule(partial(setattr, self.state, "header_text", label))

        return self._element(da, meta, title=model)

    def _diff_cb(self, model, **_):
        if self._torn_down:
            return self._placeholder(model, kind="diff")

        other = self.state.diff_pairs.get(model)
        if not other:
            with self._fields_lock:
                self._fields.pop(("diff", model), None)
            return self._placeholder(f"{model} - no difference selected",
                                     kind="diff")

        status = self.state.diff_status.get(model, "")
        if status == "computing":
            return self._placeholder(
                f"{model} minus {other} - computing", computing=True,
                kind="diff")
        if status.startswith("error"):
            return self._placeholder(f"{model} minus {other} - {status}",
                                     kind="diff")

        # Not on disk yet: _ensure_diff_async is (or is about to start)
        # computing it, and will flip diff_status - re-running this - when
        # it lands. Without this, the render triggered by diff_pairs
        # changing gets here before that "computing" status does, and
        # load_diff_field computes the whole difference itself, on the
        # Bokeh server thread - freezing the app for the full run, so no
        # busy indicator beyond the header spinner could ever reach the
        # browser.
        if not self._diff_exists(model, other):
            return self._placeholder(
                f"{model} minus {other} - computing", computing=True,
                kind="diff")

        try:
            da, meta = load_diff_field(
                self.model_dirs[model],
                self.model_dirs[other],
                model,
                other,
                self.state.variable,
                self.state.level,
                self.state.time_index,
            )
        except Exception as exc:
            traceback.print_exc()
            with self._fields_lock:
                self._fields.pop(("diff", model), None)
            return self._placeholder(
                f"{model} minus {other} - {type(exc).__name__}: {exc}",
                kind="diff")

        with self._fields_lock:
            self._fields[("diff", model)] = (da, meta)

        return self._element(da, meta, title=f"{model} minus {other}",
                             cmap=DIFF_CMAP, kind="diff")

    # -- cursor readout --------------------------------------------------

    @staticmethod
    def _sample(da, meta, x, y):
        """Nearest-neighbour lookup at a lon/lat position."""
        try:
            return float(
                da.sel({meta.lon_dim: x, meta.lat_dim: y},
                       method="nearest").item())
        except Exception:
            return None

    def _on_pointer(self, kind, x=None, y=None):
        """Update the readout from a cursor position over a panel.

        `kind` is "field" or "diff", carried in by the per-panel partial -
        it decides whether the readout reports model values or differences,
        so hovering a diff panel shows deltas rather than absolute values.

        Off-plot moves arrive as x=None; the last readout is left in place
        rather than blanked, since a value that vanishes whenever the cursor
        crosses a panel gap is more distracting than a slightly stale one.
        """
        if x is None or y is None or self._torn_down:
            return

        now = time.monotonic()
        if now - self._last_pointer < POINTER_MIN_INTERVAL:
            return
        self._last_pointer = now

        with self._fields_lock:
            entries = [(m, self._fields.get((kind, m))) for m in self.models]

        rows = []
        for model, entry in entries:
            if entry is None:
                continue
            da, meta = entry
            # Over a difference panel the values are deltas; the label says
            # which pair, so they can't be mistaken for model values.
            label = model
            if kind == "diff":
                label = f"{model} \u2212 {self.state.diff_pairs.get(model)}"
            rows.append((label, _fmt(self._sample(da, meta, x, y))))

        if not rows:
            return

        self.state.readout_rows = (
            (("Lat", _fmt_lat(y)), ("Lon", _fmt_lon(x))),
            tuple(rows),
        )

    @staticmethod
    def _readout_html(stamp, variable, level, left, right, status=None):
        """The header above the grid: valid time and variable as a
        headline, the cursor readout, and the mouse controls - with a
        banner when a difference is being computed.

        Everything here is sized to be read at a glance: the readout is the
        only way to get a number off the maps, and the mouse controls are
        the only hint that the maps can be zoomed and the colorbars
        dragged, so neither can be small grey print.

        The banner exists because a difference over a full suite takes long
        enough that silence reads as failure. The busy spinner in the
        template header is too small and too far from where the user is
        looking; this sits directly above the panel that is waiting.

        Alignment is the whole point of the readout grid. A plain inline
        run of spans reflows on every mouse move, because "0.0005791" and
        "-0.006" are different widths - so the labels visibly slide around
        and the readout is unreadable while the cursor moves. Two things fix
        that: a CSS grid with `max-content` label columns (label positions
        are set by the widest label once and then never move), and
        right-aligned values in a fixed-width column with tabular-nums and
        a monospace face, so decimal points and minus signs line up.
        """
        computing = sorted(m for m, s in (status or {}).items()
                           if s == "computing")
        errored = sorted(m for m, s in (status or {}).items()
                         if str(s).startswith("error"))

        left, right = list(left), list(right)
        n = max(len(left), len(right))
        left += [("", "")] * (n - len(left))
        right += [("", "")] * (n - len(right))

        cells = []
        for (l_lab, l_val), (r_lab, r_val) in zip(left, right):
            cells.append(
                f"<div class='rl'>{l_lab}{':' if l_lab else ''}</div>"
                f"<div class='rv'>{l_val}</div>"
                f"<div class='rl'>{r_lab}{':' if r_lab else ''}</div>"
                f"<div class='rv rv-model'>{r_val}</div>"
            )
        # The valid time heads the readout box, so the time and the values
        # read at that time are one block.
        time_row = f"<div class='readout-time'>{stamp}</div>" if stamp else ""
        readout = (
            f"<div class='readout'>{time_row}{''.join(cells)}</div>"
            if cells else
            f"<div class='readout-empty'>{time_row}"
            "<div>Hover over a map to read "
            "values here.</div></div>"
        )

        headline = ""
        if variable:
            long_name = variable_long_name(variable) or variable
            where = f" at {level} hPa" if level else ""
            headline = (
                f"<div class='readout-var'>{escape(long_name)}{where} "
                f"<span class='var-code'>({escape(variable)})</span></div>"
            )

        banner = ""
        if computing:
            banner = (
                "<div class='readout-banner readout-busy'>"
                "<span class='readout-spin'>\u27f3</span>"
                f"Computing difference for {', '.join(computing)} - "
                "a full suite can take a minute.</div>"
            )
        elif errored:
            banner = (
                "<div class='readout-banner readout-err'>"
                f"Difference failed for {', '.join(errored)} - "
                "see the panel title.</div>"
            )

        return (
            "<style>"
            ".readout-wrap{text-align:left;}"
            # The readout and the mouse controls share a row of their own,
            # below the headline and time, and stretch to the same height so
            # the two boxes' tops and bottoms line up.
            ".readout-row{display:flex;align-items:stretch;gap:28px;}"
            ".readout-var{font-size:20px;font-weight:700;color:#091422;"
            "white-space:nowrap;margin-bottom:6px;}"
            ".readout-var .var-code{font-weight:400;color:#555;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            "font-size:17px;}"
            # Spans all four readout columns. Left-aligned explicitly for
            # the same reason as .rl below.
            ".readout-time{grid-column:1/-1;font-size:15px;font-weight:600;"
            "color:#ffffff;white-space:nowrap;text-align:left;"
            "font-variant-numeric:tabular-nums;margin-bottom:2px;}"
            ".readout{display:grid;"
            # max-content sizes each label column to its widest member and
            # holds it; the fixed value columns mean a value gaining a digit
            # cannot push the next label sideways.
            "grid-template-columns:max-content 6em max-content 8.5em;"
            "column-gap:12px;row-gap:2px;font-size:17px;"
            # Header navy and pale blue (app_layout.py's header_background
            # and header link colour), so this and the mouse controls read
            # as part of the app rather than as notices pasted onto it.
            "width:max-content;background:#091422;color:#DFEFF6;"
            "border-radius:6px;padding:6px 12px;}"
            # text-align must be explicit on BOTH classes: without it the
            # cells inherit whatever alignment the surrounding Panel card
            # applies, which right-aligns the labels against each other.
            ".readout .rl{font-weight:600;text-align:left;justify-self:start;"
            "color:#DFEFF6;}"
            ".readout .rv{text-align:right;justify-self:stretch;"
            # nowrap matters for the exponent cases that survive _fmt: a
            # value wider than its column would wrap to a second line and
            # push every row below it down.
            "white-space:nowrap;font-variant-numeric:tabular-nums;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
            ".readout .rv-model{font-weight:700;color:#8fd3ff;}"
            ".readout-empty{font-size:15px;color:#DFEFF6;font-style:italic;"
            "background:#091422;border-radius:6px;"
            "padding:8px 12px;width:max-content;"
            "display:flex;flex-direction:column;justify-content:center;}"
            ".readout-empty .readout-time{font-style:normal;}"
            ".readout-hint{display:grid;grid-template-columns:max-content "
            "max-content;column-gap:10px;row-gap:4px;font-size:14px;"
            "color:#DFEFF6;white-space:nowrap;text-align:left;"
            "background:#091422;border-radius:6px;"
            "padding:8px 14px;align-content:center;}"
            ".readout-hint .hint-title{grid-column:1/3;font-weight:700;"
            "font-size:15px;color:#ffffff;}"
            ".readout-hint .hint-label{font-weight:700;color:#8fd3ff;}"
            ".readout-banner{font-size:14px;padding:6px 12px;"
            "border-radius:4px;margin-bottom:6px;width:max-content;}"
            ".readout-busy{background:#fff3cd;border:1px solid #ffc107;}"
            ".readout-err{background:#f8d7da;border:1px solid #dc3545;}"
            "@keyframes readout-spin{to{transform:rotate(360deg);}}"
            ".readout-spin{display:inline-block;animation:"
            "readout-spin 1.1s linear infinite;margin-right:6px;}"
            "</style>"
            f"{banner}"
            "<div class='readout-wrap'>"
            f"{headline}"
            "<div class='readout-row'>"
            f"{readout}"
            "<div class='readout-hint'>"
            "<div class='hint-title'>\U0001f5b1 Mouse controls</div>"
            "<div class='hint-label'>Map:</div>"
            "<div>scroll to zoom &middot; drag to pan &middot; "
            "double-click to reset</div>"
            "<div class='hint-label'>Colorbar:</div>"
            "<div>drag to rescale &middot; double-click to reset</div>"
            "</div>"
            "</div>"
            "</div>"
        )

    # -- layout ---------------------------------------------------------

    def layout(self):
        """The hv.Layout. Built once; the DynamicMaps update in place."""
        if self._layout is not None:
            return self._layout

        panels = []
        for i, model in enumerate(self.models):
            field = hv.DynamicMap(
                partial(self._field_cb, model, i == 0),
                streams=[self._field_stream])
            diff = hv.DynamicMap(
                partial(self._diff_cb, model),
                streams=[self._diff_stream])

            # rasterize() must wrap the DynamicMap rather than being called
            # inside the callback - it is itself a dynamic operation and
            # cannot be returned from one.
            #
            # Styling is applied downstream of rasterize via .apply.opts with
            # param references, so changing the colormap or colour limits
            # restyles the existing render instead of re-reading the field.
            #
            # The diff branch carries only clim here. Its colormap is set on
            # the element instead: mixing a static value with param
            # references in this chain makes HoloViews rebuild the branch in
            # a way that drops the element-level opts.
            field = rasterize(field, precompute=True).apply.opts(
                clim=self.state.param.field_clim,
                cmap=self.state.param.cmap,
            )
            diff = rasterize(diff, precompute=True).apply.opts(
                clim=self.state.param.diff_clim,
            )

            # One PointerXY per panel, sourced from the object that actually
            # gets rendered. The `kind` bound into the subscriber is how the
            # readout knows whether the cursor is over a field or a diff.
            for obj, kind in ((field, "field"), (diff, "diff")):
                st = hv.streams.PointerXY(x=None, y=None, source=obj)
                st.add_subscriber(partial(self._on_pointer, kind))
                self._pointer_streams.append(st)

            panels += [field, diff]

        # toolbar=None removes the button STRIP, not the tools. Pan,
        # wheel-zoom and reset stay attached and active with no UI to turn
        # them off - which is the point: all three are always on, so the
        # buttons were only an opportunity to break navigation by accident.
        #
        # This has to be here rather than in _panel_opts: toolbar is a
        # Layout-level option, and setting it on an element neither
        # suppresses the strip nor leaves tool activation alone.
        #
        # shared_axes is what links the panels. It links them PAIRWISE, not
        # all-to-all - see the module docstring.
        #
        # No sizing_mode: the panels are fixed-size, so the gridplot takes
        # its natural width from them. stretch_width here would clamp it to
        # the window again and reintroduce the clipping PANEL_FRAME_WIDTH
        # exists to avoid.
        self._layout = hv.Layout(panels).cols(2).opts(
            shared_axes=True,
            toolbar=None,
        )
        return self._layout

    def panel(self):
        """Valid time and cursor readout above the grid."""
        # Bound to the timestamp, the rows AND diff_status, so a slider tick
        # refreshes the time even though the cursor hasn't moved, a cursor
        # move refreshes the values without dropping the time, and a
        # difference computation starting or finishing raises or clears the
        # banner without waiting for either. Before the cursor first enters
        # a panel, readout_rows is empty and this renders the timestamp and
        # the navigation hint alone.
        readout = pn.pane.HTML(
            pn.bind(lambda stamp, var, level, rows, status: self._readout_html(
                        stamp, var, level, rows[0], rows[1], status),
                    self.state.param.header_text,
                    self.state.param.variable,
                    self.state.param.level,
                    self.state.param.readout_rows,
                    self.state.param.diff_status),
            margin=(4, 0, 8, 12),
            # Reserves the readout's rows so the plots below don't shift
            # down the first time the cursor enters a panel.
            min_height=128,
        )

        # No in-card busy state here: while a difference computes, the app
        # lays a spinner over the whole Visualization tab instead (see
        # _diff_busy_overlay in app_layout.py), which also blocks
        # interaction with the tab until it finishes.
        self._hv_pane = pn.pane.HoloViews(self.layout())
        return pn.Column(
            readout,
            self._hv_pane,
            sizing_mode="fixed",
        )

    def card(self, title="Forecast fields", **kwargs):
        # Sized to its content, not stretched to the window: the grid inside
        # is fixed-size, and a window-width card squeezes the gridplot's CSS
        # grid columns below the figures' width, clipping each map and its
        # colorbar inside its cell. That means sizing_mode="fixed" here AND
        # on every layout between this and the grid (see panel()) - Panel
        # infers stretch_width on a parent from any stretch_width child, so
        # one stretched descendant is enough to undo it. min-width keeps the
        # card filling the main area when the grid is the narrower of the two.
        #
        # width:max-content is what actually sizes it. sizing_mode="fixed"
        # with no width leaves the width to the browser, and Chrome (unlike
        # Firefox) resolves that to the container's width, not the grid's.
        opts = dict(collapsible=False, sizing_mode="fixed",
                    styles={"width": "max-content", "min-width": "100%"})
        opts.update(kwargs)
        return pn.Card(self.panel(), title=title, **opts)

    # -- difference selectors (sidebar) ---------------------------------

    def diff_selectors(self, width=200):
        """Column of one Select per model, for the sidebar."""
        def options(model):
            # Pairs whose difference is already saved with the suite (from
            # this session or an earlier one) are marked, since picking
            # one shows it straight away rather than computing it.
            opts = {"None": "None"}
            for other in self.models:
                if other != model:
                    saved = self._diff_exists(model, other)
                    opts[f"{other} (saved)" if saved else other] = other
            return opts

        widgets = {}
        for model in self.models:
            widgets[model] = pn.widgets.Select(
                name=f"{model} minus",
                options=options(model),
                value="None",
                width=width,
            )

        def sync(*_):
            # Rebinding, not mutating: param only fires on assignment, so an
            # in-place dict update would silently fail to trigger a redraw.
            pairs = {
                m: (None if w.value == "None" else w.value)
                for m, w in widgets.items()
            }
            self.state.diff_pairs = pairs
            for a, b in pairs.items():
                if b:
                    self._ensure_diff_async(a, b)

        def on_status(event):
            """Reflect computation state on the control the user just used.

            Disabling is not only a signal: it also stops a second pair
            being queued while the first is still running, which would put
            two multi-gigabyte diffs on the box at once.
            """
            for model, w in widgets.items():
                busy = event.new.get(model) == "computing"
                w.disabled = busy
                w.name = (f"{model} minus  (computing\u2026)" if busy
                          else f"{model} minus")
                if event.new.get(model) == "ready":
                    w.options = options(model)

        for w in widgets.values():
            w.param.watch(sync, "value")
        self._status_watcher = self.state.param.watch(on_status, "diff_status")
        sync()

        self._diff_widgets = widgets
        return pn.Column("### Differences", *widgets.values())

    # -- background computation -----------------------------------------

    def _diff_exists(self, a, b):
        return self._diff_path(a, b).exists()

    def _diff_path(self, a, b):
        """(a minus b)'s file, saved in a's directory with the suite."""
        return diff_file_path(self.model_dirs[a], a, b)

    def _set_status(self, model, value):
        def apply():
            self.state.diff_status = {**self.state.diff_status, model: value}
        _schedule(apply)

    def _ensure_diff_async(self, a, b):
        """Compute a model pair's difference off the event loop.

        Diffing two full forecast suites takes long enough that doing it
        inside the DynamicMap callback would block the Bokeh server thread
        and freeze the whole app - including the panels that have nothing to
        do with this pair. The callback only ever reads an already-computed
        file; this does the work and flips diff_status when it lands.
        """
        if self._diff_exists(a, b):
            self._set_status(a, "ready")
            # Refresh the diff clim even on a cache hit. Without this, a
            # suite whose diffs were computed in an earlier session never
            # gets diff_clim set at all: the initial refresh_clims runs
            # before any pair is selected and finds nothing, and this early
            # return used to be the only other path.
            self.refresh_diff_clim_async()
            return

        key = (a, b)
        with self._pending_lock:
            if key in self._pending:
                return
            self._pending.add(key)

        self._set_status(a, "computing")

        def work():
            try:
                compute_model_difference(
                    self.model_dirs[a], self.model_dirs[b], a, b,
                )
                if self._torn_down:
                    return
                self._set_status(a, "ready")
                self.refresh_diff_clim_async()
            except Exception as exc:
                traceback.print_exc()
                if not self._torn_down:
                    self._set_status(a, f"error: {exc}")
            finally:
                with self._pending_lock:
                    self._pending.discard(key)

        threading.Thread(target=work, daemon=True,
                         name=f"diff-{a}-minus-{b}").start()

    # -- colour limits ---------------------------------------------------

    def refresh_clims(self, sample_steps=3):
        """Recompute field and difference colour limits. Blocking."""
        self._refresh_field_clim(sample_steps)
        self._refresh_diff_clim(sample_steps)

    def _refresh_field_clim(self, sample_steps=3):
        if self._torn_down:
            return

        # An explicit non-zero min/max from the sidebar wins outright.
        if self.state.cmap_min != 0.0 or self.state.cmap_max != 0.0:
            _schedule(partial(setattr, self.state, "field_clim",
                              (self.state.cmap_min, self.state.cmap_max)))
            return

        key = self._field_clim_key()
        lo, hi = np.inf, -np.inf
        for model in self.models:
            try:
                a, b = field_range(self.model_dirs[model], self.state.variable,
                                   self.state.level, sample_steps)
            except Exception:
                traceback.print_exc()
                continue
            lo, hi = min(lo, a), max(hi, b)

        if np.isfinite(lo) and np.isfinite(hi) and not self._torn_down:
            self._auto_clims["field"] = (key, (lo, hi))
            _schedule(partial(setattr, self.state, "field_clim", (lo, hi)))

    def _refresh_diff_clim(self, sample_steps=3):
        if self._torn_down:
            return

        key = self._diff_clim_key()
        m = 0.0
        for a, b in self.state.diff_pairs.items():
            if not b or not self._diff_exists(a, b):
                continue
            try:
                lo, hi = symmetric_diff_range(
                    self.model_dirs[a], self.model_dirs[b], a, b,
                    self.state.variable, self.state.level, sample_steps,
                )
            except Exception:
                traceback.print_exc()
                continue
            m = max(m, abs(lo), abs(hi))

        # A shared symmetric range across all difference panels means the
        # panels are directly comparable, and coolwarm's white sits exactly
        # at zero. With an asymmetric range an unbiased field reads as
        # biased, which is actively misleading on a difference plot.
        clim = (-m, m) if m > 0 else CLIM_UNSET
        if m > 0:
            self._auto_clims["diff"] = (key, clim)
        if not self._torn_down:
            _schedule(partial(setattr, self.state, "diff_clim", clim))

    def refresh_clims_async(self, sample_steps=3):
        threading.Thread(
            target=self.refresh_clims, args=(sample_steps,),
            daemon=True, name="clim-refresh").start()

    def refresh_field_clim_async(self, sample_steps=3):
        threading.Thread(
            target=self._refresh_field_clim, args=(sample_steps,),
            daemon=True, name="field-clim-refresh").start()

    def refresh_diff_clim_async(self, sample_steps=3):
        threading.Thread(
            target=self._refresh_diff_clim, args=(sample_steps,),
            daemon=True, name="diff-clim-refresh").start()

    # -- wiring ----------------------------------------------------------

    def set_time_bounds(self, n_steps):
        """Record the forecast length and clamp the current index into it.

        Named for what it used to do. It no longer touches
        param.time_index.bounds - partly because those bounds are gone (see
        PlotGridState.time_index), and partly because the old
        `self.state.param.time_index.bounds = ...` reached the CLASS-level
        Parameter object rather than this instance's.
        """
        self.n_steps = max(1, int(n_steps))
        if self.state.time_index > self.n_steps - 1:
            self.state.time_index = self.n_steps - 1

    def frame_spec(self):
        """What the video exporter needs to render frames itself.

        The grid is Bokeh-rendered client-side, so there is no server-side
        image to capture - the exporter reconstructs each frame through
        earth2StudioPlot.plot_e2s_field instead. Panels are returned in
        display order; each entry feeds
        plot_e2s_field(dir, variable, level, t, cmap=cmap,
                       vmin=clim[0], vmax=clim[1]).
        """
        panels = []
        for model in self.models:
            panels.append({
                "kind": "field",
                "title": model,
                "dir": self.model_dirs[model],
                "cmap": self.state.cmap,
                "clim": self.state.field_clim,
            })
            other = self.state.diff_pairs.get(model)
            if other and self._diff_exists(model, other):
                panels.append({
                    "kind": "diff",
                    "title": f"{model} minus {other}",
                    "dir": self._diff_path(model, other),
                    "cmap": DIFF_CMAP,
                    "clim": self.state.diff_clim,
                })
        return {
            "panels": panels,
            "variable": self.state.variable,
            "level": self.state.level,
            "n_steps": self.n_steps,
            "ncols": 2,
            "boundaries": self.state.boundaries,
        }
