"""
visualization/videoExport.py

Export the Visualization tab's current rendering as an MP4, sweeping the
shared time index from the first forecast step to the last.

Frames are produced by calling plot_e2s_field directly -- the same function
that backs the on-screen panes -- rather than by driving the live widgets, so
the export never fights the session for control of the plot.

The user-facing knob is "frames per second": how many forecast time steps
are shown per second of playback. That is deliberately distinct from the MP4
container frame rate. ffmpeg reads the PNG sequence at the user's rate
(-framerate) and duplicates frames up to the output rate (-r), so 2 frames/s
still yields a smooth 30 fps file rather than a 2 fps one that some players
stutter on.

The finished MP4 is saved in the simulation suite's own directory when the
user owns it, otherwise in the user's scratch directory (so exports of a
shared suite such as ExampleDataset don't land in someone else's space),
and then handed to the browser as a download.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import panel as pn
import param
from PIL import Image, ImageDraw

from cf_convert import variable_long_name
from dimensions import level_text
from visualization.earth2StudioPlot import load_e2s_field, plot_e2s_field
from usage_log import log_event, short_error

DEFAULT_FRAMES_PER_SECOND = 2.0
MP4_FRAME_RATE = 30

# The composited frame's own size depends entirely on the current layout --
# panel count, rows, colorbar tick-label width -- and rarely lands anywhere
# near a standard video resolution or aspect ratio on its own. Scaling to a
# named target (letterboxed to preserve the frame's aspect ratio -- see
# _worker) makes the output predictable to share/embed regardless of what's
# on screen. 1080p is kept as the default: it's still the most universally
# expected resolution for a shared MP4, even though the source frame is
# usually a wider aspect ratio than 16:9 and ends up letterboxed top/bottom.
RESOLUTION_SOURCE = "Source resolution (no scaling)"
RESOLUTIONS = {
    "720p (1280×720)": (1280, 720),
    "1080p (1920×1080)": (1920, 1080),
    "1440p (2560×1440)": (2560, 1440),
    "4K / 2160p (3840×2160)": (3840, 2160),
    RESOLUTION_SOURCE: None,
}
DEFAULT_RESOLUTION = "1080p (1920×1080)"

# How many simulations sit side by side in the video. The screen stacks one
# per row, which leaves most of a landscape video empty once there are two
# or more; "best fit" picks whichever count fills the target frame most (see
# _best_per_row). A simulation's difference card always stays beside it.
LAYOUT_AUTO = "Best fit (auto)"
LAYOUTS = {
    LAYOUT_AUTO: None,
    "1 simulation per row (as on screen)": 1,
    "2 simulations per row": 2,
    "3 simulations per row": 3,
    "4 simulations per row": 4,
}
DEFAULT_LAYOUT = LAYOUT_AUTO
# What "best fit" aims for when exporting at source resolution, which has no
# target frame of its own: the shape most players and screens expect.
_FALLBACK_ASPECT = (16, 9)

_PREVIEW_W = 420
_PREVIEW_BAND = (222, 222, 222)   # the part of the video left empty

_PAD = 10
_LABEL_H = 30
_HEADER_H = 92
_BG = (255, 255, 255)
_INK = (20, 20, 20)
_INK_SOFT = (70, 70, 70)

# plot_e2s_field renders through matplotlib's pyplot state machine, which is
# not thread safe. The export worker must not race the live session's own
# renders, or frames come back garbled/blank.
_RENDER_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _output_dir(suite_dir: Path) -> Path:
    """Where to save the MP4: the suite's own directory if the user owns
    it, otherwise their scratch directory. Shared suites (ExampleDataset,
    or one loaded from a colleague's space) would otherwise collect every
    user's exports in the owner's directory."""
    try:
        if suite_dir.stat().st_uid == os.getuid():
            return suite_dir
    except OSError:
        pass
    return Path(f"/glade/derecho/scratch/{os.environ['USER']}")


def _ffmpeg_exe() -> str:
    """Locate a working ffmpeg.

    Prefers imageio-ffmpeg's statically-linked binary, which carries its own
    libx264 and so is immune to the conda env's shared-library state -- the
    conda-forge ffmpeg in creditJun3 is a stale 2.8.6 build that fails at
    load time looking for libx264.so.138 against an installed .so.164.
    Whichever candidate is chosen is verified by actually running it, since
    a path from shutil.which proves only that a file exists, not that it
    can link.
    """
    candidates = []
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)

    problems = []
    for exe in candidates:
        try:
            proc = subprocess.run([exe, "-version"], capture_output=True,
                                  text=True, timeout=20)
            if proc.returncode == 0:
                return exe
            first = (proc.stderr.strip().splitlines() or ["non-zero exit"])[0]
            problems.append(f"{exe}: {first}")
        except Exception as e:
            problems.append(f"{exe}: {e}")

    detail = ("\n" + "\n".join(problems)) if problems else ""
    raise RuntimeError(
        "No working ffmpeg found. Install the self-contained build with "
        "`pip install imageio-ffmpeg` in the active env." + detail
    )


def _ui(fn, *args, **kwargs):
    """Apply a widget mutation on the Bokeh document thread.

    Worker threads must not touch widget state directly; pn.state.execute
    defers the call onto the session's event loop. Falls back to a direct
    call when there is no live session (headless testing).
    """
    try:
        pn.state.execute(lambda: fn(*args, **kwargs))
    except Exception:
        fn(*args, **kwargs)


@lru_cache(maxsize=8)
def _font(size, bold=False):
    """A real TrueType font at `size`, falling back to PIL's bitmap default.

    PIL's load_default() ignores size entirely, so labels would come out
    tiny on a 1200px-wide composite; matplotlib always ships DejaVu Sans,
    so findfont is a reliable source for a scalable face.
    """
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        face = font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal")
        return ImageFont.truetype(font_manager.findfont(face), size)
    except Exception:
        return ImageFont.load_default()


class _Tile:
    """One panel of the composite frame, with a color range fixed for the
    whole sweep.

    Mirrors one card of the on-screen grid: either a model's field or a
    model-minus-model difference. The interactive plot re-derives its range
    on every step, which is fine when scrubbing but makes a video's colorbar
    breathe and its field flicker, so the range is resolved once up front and
    then held.

    Field panels arrive with vmin/vmax already fixed -- the exact range the
    GUI is currently showing (frame_spec()'s clim for a PlotGrid, or the
    Colormap Min/Max boxes for a flat dataset), so there is nothing left to
    scan for. Difference panels have no such GUI-visible range to copy (the
    on-screen diff recomputes a fresh auto range every time step), so vmin/
    vmax start as None and get resolved by scan()/finalize() below.
    """

    def __init__(self, label, model_dir, is_diff, cmap, vmin=None, vmax=None,
                 boundaries=None):
        self.label = label
        self.boundaries = boundaries   # BOUNDARY_OPTIONS label, or None
        self.model_dir = model_dir
        self.is_diff = is_diff
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self._scan_lo = None
        self._scan_hi = None
        self.size = None      # fixed at the first rendered frame; libx264
                              # requires constant dimensions

    @property
    def needs_scan(self):
        """True for a range that hasn't already been resolved elsewhere."""
        return self.vmin is None or self.vmax is None

    def scan(self, var_name, level, t):
        """Record this step's data extrema without keeping the image.

        plot_e2s_field reports the range it actually used, so an auto-ranged
        call doubles as a probe. There is no cheaper path to the extrema
        through the current plotting interface.
        """
        _, lo, hi = plot_e2s_field(
            model_dir=self.model_dir, base_or_var=var_name, level=level, t=t,
            cmap=self.cmap, vmin=None, vmax=None,
        )
        self._scan_lo = lo if self._scan_lo is None else min(self._scan_lo, lo)
        self._scan_hi = hi if self._scan_hi is None else max(self._scan_hi, hi)

    def finalize(self):
        """Resolve the fixed range from whatever scanning found.

        No-op for a tile that already arrived with a fixed range. Scanning
        the FULL series matters most for a difference tile: at t=0 both
        models start from the same analysis, so the difference is nothing
        but floating-point noise (~1e-10). Locking that as the range makes
        every later step -- where genuine divergence is many orders of
        magnitude larger -- saturate to solid red and blue.
        """
        if self.vmin is None or self.vmax is None:
            lo = self._scan_lo if self._scan_lo is not None else 0.0
            hi = self._scan_hi if self._scan_hi is not None else 0.0
            if self.is_diff:
                # Symmetric about zero so coolwarm's neutral midpoint lands
                # on zero difference -- same reasoning as
                # DatasetPlot2._render_diff.
                max_abs = max(abs(lo), abs(hi))
                if max_abs == 0.0:
                    max_abs = 1e-12   # a degenerate all-zero field: avoid vmin==vmax
                self.vmin, self.vmax = -max_abs, max_abs
            else:
                self.vmin, self.vmax = lo, hi
        if self.vmin == self.vmax:
            # A pinned or GUI-displayed range can arrive already degenerate
            # (e.g. both boxes at 0.0) -- nudge it too, not just a scanned one.
            self.vmin, self.vmax = self.vmin - 1e-12, self.vmax + 1e-12

    def render(self, var_name, level, t):
        """Return a PIL image of this panel at time step t."""
        buf, _, _ = plot_e2s_field(
            model_dir=self.model_dir, base_or_var=var_name, level=level, t=t,
            cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
            boundaries=self.boundaries,
        )
        img = Image.open(buf).convert("RGB")
        if self.size is None:
            self.size = img.size
        elif img.size != self.size:
            # Colorbar tick labels can change width between steps (e.g. "5"
            # vs "-12.5"), nudging the figure's pixel dimensions. H.264
            # requires every frame to be identical in size.
            img = img.resize(self.size, Image.LANCZOS)
        return img


def _header_lines(var, level, level_dim, units, meta, t, total):
    """The two header lines above every frame: what the field is, then when.

    Mirrors the Visualization tab's readout headline -- long name, level and
    short name -- with the units added, since a video has no hover readout
    or colorbar title to carry them.
    """
    long_name = variable_long_name(var) or var
    where = f" at {level_text(level, level_dim)}" if level else ""
    title = f"{long_name}{where} ({var})"
    if units:
        title += f"  [{units}]"

    when = []
    if meta is not None:
        if meta.valid_time is not None:
            when.append(f"Valid {meta.time_label()}")
        if meta.lead_hours is not None:
            when.append(f"Forecast +{meta.lead_hours:.0f}h")
    when.append(f"Step {t + 1} of {total}")
    return title, "   \u00b7   ".join(when)


def _arrange(units, per_row):
    """Regroup per-simulation units into frame rows of `per_row` units.

    A unit is one simulation's cards -- its field, plus its difference card
    when active -- so the two are never split across rows.
    """
    return [[card for unit in units[i:i + per_row] for card in unit]
            for i in range(0, len(units), per_row)]


def _layout_dims(rows):
    """(row_dims, width, height) of the frame _compose would build.

    Needs only each image's size, so a layout can be sized before anything
    is drawn.
    """
    row_dims = []
    for row in rows:
        w = sum(im.width for _, im in row) + _PAD * (len(row) + 1)
        h = max(im.height for _, im in row) + _LABEL_H + _PAD
        row_dims.append((w, h))

    width = max(w for w, _ in row_dims)
    height = _HEADER_H + sum(h for _, h in row_dims) + _PAD
    # libx264 with yuv420p chroma subsampling requires even dimensions.
    width += width % 2
    height += height % 2
    return row_dims, width, height


def _fit_scale(rows, target):
    """How much a frame laid out as `rows` is scaled to fit `target`."""
    tw, th = target or _FALLBACK_ASPECT
    _, w, h = _layout_dims(rows)
    return min(tw / w, th / h)


def _best_per_row(units, target):
    """The simulations-per-row count that shows the panels largest in
    `target` -- i.e. leaves the least of the video empty. Ties go to fewer
    per row, the closer match to the screen."""
    best, best_scale = 1, 0.0
    for n in range(1, len(units) + 1):
        scale = _fit_scale(_arrange(units, n), target)
        if scale > best_scale * (1 + 1e-6):
            best, best_scale = n, scale
    return best


def _resolve_per_row(layout, units, target):
    n = LAYOUTS.get(layout)
    return min(n, len(units)) if n else _best_per_row(units, target)


def _compose(rows, title, subtitle):
    """Tile rendered panels into one frame.

    `rows` is a list of rows, each a list of (label, PIL.Image) -- see
    _arrange. Rows are centred, so a short last row sits in the middle.
    `title` and `subtitle` are centred above the panels (see _header_lines).
    """
    row_dims, width, height = _layout_dims(rows)

    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 10), title, fill=_INK,
              font=_font(30, bold=True), anchor="ma")
    draw.text((width // 2, 52), subtitle, fill=_INK_SOFT,
              font=_font(22), anchor="ma")

    y = _HEADER_H
    for row, (row_w, row_h) in zip(rows, row_dims):
        x = (width - row_w) // 2 + _PAD
        for label, im in row:
            draw.text((x + im.width // 2, y + 6), label,
                      fill=_INK, font=_font(19), anchor="ma")
            canvas.paste(im, (x, y + _LABEL_H))
            x += im.width + _PAD
        y += row_h
    return canvas


def _preview(frame, target):
    """A small picture of the finished video frame, and the share of it
    the panels fill.

    The letterbox bands are drawn grey rather than the video's white, so
    the empty space a layout leaves is plain to see.
    """
    fw, fh = frame.size
    if target is None:
        # Source resolution: the video is exactly the frame, no bands.
        tw, th, scale = fw, fh, 1.0
    else:
        tw, th = target
        scale = min(tw / fw, th / fh)
    k = _PREVIEW_W / tw
    canvas = Image.new("RGB", (_PREVIEW_W, max(1, round(th * k))), _PREVIEW_BAND)
    cw, ch = max(1, round(fw * scale * k)), max(1, round(fh * scale * k))
    canvas.paste(frame.resize((cw, ch), Image.LANCZOS),
                 ((canvas.width - cw) // 2, (canvas.height - ch) // 2))
    ImageDraw.Draw(canvas).rectangle(
        (0, 0, canvas.width - 1, canvas.height - 1), outline=_INK_SOFT)
    fill = (fw * scale) * (fh * scale) / (tw * th)
    return canvas, fill


class BrowserDownload(pn.custom.JSComponent):
    """Saves a file to the user's machine without a second click.

    FileDownload only starts a download when its own button is clicked, so
    it can't follow on from an export that finishes in the background.
    Setting `data` (base64) here makes the browser save it as `filename`
    straight away. `filename` must be set before `data`. The browser clears
    `data` once saved, so the session doesn't keep a copy of the video.
    """

    data = param.String(default="")
    filename = param.String(default="download")
    mime = param.String(default="application/octet-stream")

    _esm = """
    export function render({ model }) {
      model.on("data", () => {
        if (!model.data) return;
        const bin = atob(model.data);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const url = URL.createObjectURL(new Blob([bytes], { type: model.mime }));
        const a = document.createElement("a");
        a.href = url;
        a.download = model.filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        model.data = "";
      });
    }
    """


# --------------------------------------------------------------------------
# modal
# --------------------------------------------------------------------------

class VideoExportPanel(param.Parameterized):
    """Export Video button + options modal.

    Follows the LoadSuiteDialog convention: exposes `.open_button` and
    `.modal`, both of which must be appended to the sidebar Column.
    """

    frames_per_second = param.Number(
        default=DEFAULT_FRAMES_PER_SECOND,
        bounds=(0.25, 24.0),
        step=0.25,
        label="Frames per second",
        doc="How many forecast time steps are shown per second of playback.",
    )

    resolution = param.Selector(
        default=DEFAULT_RESOLUTION,
        objects=list(RESOLUTIONS),
        label="Output resolution",
        doc="Target MP4 resolution. The composited frame is letterboxed to "
            "fit -- its own aspect ratio depends on the layout and rarely "
            "matches the target's exactly.",
    )

    layout = param.Selector(
        default=DEFAULT_LAYOUT,
        objects=list(LAYOUTS),
        label="Layout",
        doc="How many simulations sit side by side in the video.",
    )

    # True while an export runs, for the tab's busy spinner.
    busy = param.Boolean(default=False, precedence=-1)

    def __init__(self, controls, active_plot_fn, suite_dir_fn, **params):
        """`suite_dir_fn` returns the directory of the suite on screen,
        which is where the finished MP4 is saved if the user owns it."""
        super().__init__(**params)
        self.controls = controls
        self._active_plot_fn = active_plot_fn
        self._suite_dir_fn = suite_dir_fn
        self._cancel = threading.Event()
        self._thread = None

        # Sized/margined to match load_suite_dialog.open_button exactly, so
        # both stretch to the same effective width inside the 250px sidebar.
        self.open_button = pn.widgets.Button(
            name="Download Video",
            button_type="primary",
            sizing_mode="stretch_width",
            margin=(10, 10, 0, 0),
        )
        self.open_button.on_click(self._open)

        self._rps = pn.widgets.FloatSlider.from_param(
            self.param.frames_per_second, sizing_mode="stretch_width",
        )
        # One checkbox per simulation (model) on screen, rebuilt each time
        # the dialog opens - see _build_model_checks. _order is the order
        # they appear in the video; it's kept across opens so a user's
        # arrangement survives closing the dialog.
        self._model_checks = {}
        self._order = []
        self._models_box = pn.Column(sizing_mode="stretch_width", margin=0)
        self._resolution = pn.widgets.Select.from_param(
            self.param.resolution, sizing_mode="stretch_width",
        )
        self._layout = pn.widgets.Select.from_param(
            self.param.layout, sizing_mode="stretch_width",
        )
        # A miniature of the finished frame at the current time step, so
        # the effect of a layout is visible before committing to an export.
        # Rendered tiles are cached, so reordering or changing the layout
        # only recomposes; _preview_gen drops results a newer request has
        # superseded.
        self._preview_pane = pn.pane.PNG(None, width=_PREVIEW_W, margin=(0, 10))
        self._preview_caption = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._preview_cache = {}
        self._preview_gen = 0
        self._summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._progress = pn.indicators.Progress(
            value=0, max=100, sizing_mode="stretch_width", visible=False,
        )
        self._status = pn.pane.Markdown("", sizing_mode="stretch_width")
        # Shown once the browser has been handed the MP4; the progress
        # status alone was easy to miss.
        self._done = pn.pane.Alert(
            "", alert_type="success", sizing_mode="stretch_width",
            visible=False)
        self._browser_download = BrowserDownload(
            mime="video/mp4", width=0, height=0, margin=0)
        self._export_btn = pn.widgets.Button(
            name="Download MP4", button_type="primary", width=140)
        self._cancel_btn = pn.widgets.Button(
            name="Cancel", button_type="light", width=110, disabled=True,
        )
        self._export_btn.on_click(self._start)
        self._cancel_btn.on_click(self._request_cancel)

        self.modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown("### Export Video"),
                pn.pane.Markdown(
                    "Renders the current visualization across every forecast "
                    "time step and encodes it to MP4."
                ),
                pn.layout.Divider(),
                self._models_box,
                self._rps,
                self._resolution,
                self._layout,
                self._preview_pane,
                self._preview_caption,
                self._summary,
                pn.layout.Divider(),
                pn.Row(self._export_btn, self._cancel_btn),
                self._progress,
                self._status,
                self._done,
                self._browser_download,
                width=480,
                sizing_mode="stretch_width",
            ),
            name="export_video_modal",
            show_close_button=True,
            background_close=False,
        )

    # -- state -------------------------------------------------------------

    def _nframes(self):
        """Forecast step count, taken from the shared time slider's range.

        SharedPlotControls.update_choices already resolved the
        lead_time-vs-time distinction and clamped to the longest checked
        dataset, so there is no need to re-derive it from the metadata.
        """
        return int(self.controls.time_slider.end) + 1

    def _grid_spec(self, plot):
        """Row spec for a PlotGrid, from its purpose-built frame_spec().

        frame_spec() already resolves cmap and colour range exactly as the
        GUI currently shows them (the sidebar's Colormap Min/Max when set,
        else the grid's own auto-scan) -- there is nothing left for the
        exporter to compute for a field panel. A difference panel's clim may
        still be the not-yet-resolved sentinel if its background scan hasn't
        landed yet; that shows up as (None, None) and _Tile scans for itself.
        """
        panels = plot.frame_spec()["panels"]
        rows = []
        for p in panels:
            vmin, vmax = p["clim"]
            entry = (p["title"], p["dir"], p["kind"] == "diff", p["cmap"], vmin, vmax)
            if p["kind"] == "field":
                rows.append([entry])
            else:
                rows[-1].append(entry)
        return rows or None

    def _dataset_spec(self, plot):
        """Row spec for a flat single-dataset DatasetPlot2.

        Field panels take the exact range currently displayed in the
        Colormap Min/Max boxes -- the default auto-computed value if the
        user never typed one, or their explicit value otherwise. Difference
        panels have no such GUI-visible range (the on-screen diff recomputes
        a fresh auto range every time step), so they're left unresolved for
        _Tile to scan.
        """
        raw_rows = plot.export_panels()
        if not raw_rows:
            return None
        cmap = self.controls._colormaps.get(self.controls.colormap, "viridis")
        vmin = self.controls.cmap_min_input.value
        vmax = self.controls.cmap_max_input.value
        return [
            [
                (label, path, is_diff, "coolwarm", None, None) if is_diff
                else (label, path, is_diff, cmap, vmin, vmax)
                for label, path, is_diff in row
            ]
            for row in raw_rows
        ]

    def _panel_spec(self, selected_only=False):
        """What the Visualization tab is currently rendering, or None.

        A list of rows, each a list of (label, model_dir, is_diff, cmap,
        vmin, vmax) tuples, mirroring the on-screen layout -- one row per
        model, with its difference card alongside when active. Each row's
        first entry is the model's own field, labelled with the model name.

        With `selected_only`, rows for simulations the user unticked in the
        dialog are dropped and the rest are put in the dialog's order.
        """
        plot = self._active_plot_fn()
        if plot is None or not self.controls.var_name:
            return None
        try:
            if hasattr(plot, "frame_spec"):
                rows = self._grid_spec(plot)
            else:
                rows = self._dataset_spec(plot)
        except Exception:
            return None
        if rows and selected_only:
            rows = [r for r in rows if self._model_selected(r[0][0])]
            rank = {m: i for i, m in enumerate(self._order)}
            rows.sort(key=lambda r: rank.get(r[0][0], len(rank)))
        return rows or None

    def _model_selected(self, model):
        check = self._model_checks.get(model)
        return check is None or check.value

    def _build_model_checks(self):
        """One checkbox per simulation on screen, all ticked.

        Rebuilt on every open, since the suite (and so its models) may have
        changed since the dialog was last shown. The previous order is kept
        for models still present; new ones join at the end in screen order.
        """
        models = [row[0][0] for row in self._panel_spec() or []]
        self._order = ([m for m in self._order if m in models]
                       + [m for m in models if m not in self._order])
        self._model_checks = {}
        for model in models:
            check = pn.widgets.Checkbox(name=model, value=True,
                                        margin=(6, 10, 0, 4))
            check.param.watch(lambda _e: self._options_changed(), "value")
            self._model_checks[model] = check
        self._render_model_rows()

    def _render_model_rows(self):
        """Lay out the simulation list: ▲/▼ to reorder, checkbox to include."""
        if not self._order:
            self._models_box.objects = []
            return
        locked = self._exporting()
        last = len(self._order) - 1
        lines = []
        for i, model in enumerate(self._order):
            up = pn.widgets.Button(name="▲", button_type="light",
                                   width=34, margin=(2, 0, 2, 10),
                                   disabled=locked or i == 0)
            down = pn.widgets.Button(name="▼", button_type="light",
                                     width=34, margin=(2, 0),
                                     disabled=locked or i == last)
            up.on_click(lambda _e, m=model: self._move(m, -1))
            down.on_click(lambda _e, m=model: self._move(m, +1))
            lines.append(pn.Row(up, down, self._model_checks[model], margin=0))
        self._models_box.objects = [
            pn.pane.Markdown("**Simulations** — in video order",
                             margin=(0, 10)),
            *lines,
        ]

    def _move(self, model, step):
        i = self._order.index(model)
        j = i + step
        if self._exporting() or not 0 <= j < len(self._order):
            return
        self._order[i], self._order[j] = self._order[j], self._order[i]
        self._render_model_rows()
        self._refresh_preview()

    def _options_changed(self):
        self._refresh_summary()
        self._refresh_preview()

    def _boundaries(self):
        """The boundary lines the Visualization tab is drawing, or None.

        Taken from the plot grid's frame_spec(), so the video shows exactly
        what's on screen. A flat single dataset (DatasetPlot2) draws no
        boundaries, so its export doesn't either.
        """
        plot = self._active_plot_fn()
        if plot is None or not hasattr(plot, "frame_spec"):
            return None
        try:
            return plot.frame_spec().get("boundaries")
        except Exception:
            return None

    # -- callbacks ---------------------------------------------------------

    def _exporting(self):
        return bool(self._thread and self._thread.is_alive())

    def _open(self, _event=None):
        if not self._exporting():
            # Left alone mid-export, so reopening the dialog shows the
            # run's progress rather than a fresh form.
            self._status.object = ""
            self._progress.visible = False
            self._done.visible = False
            self._reset_open_button()
            self._build_model_checks()
            self._refresh_summary()
            self._refresh_preview()
        self.modal.open = True

    @param.depends("resolution", "layout", watch=True)
    def _refresh_preview(self):
        """Re-render the preview in the background for the current options."""
        if self._exporting():
            return
        self._preview_gen += 1
        rows = self._panel_spec(selected_only=True)
        if rows is None or self.controls.time_slider.disabled:
            self._preview_pane.object = None
            self._preview_pane.visible = False
            self._preview_caption.object = ""
            return
        snap = dict(
            var_name=self.controls.var_name,
            level=self.controls.level_value,
            level_dim=self.controls.level_dim,
            t=int(self.controls.time_slider.value),
            nframes=self._nframes(),
            boundaries=self._boundaries(),
            resolution=RESOLUTIONS[self.resolution],
            layout=self.layout,
        )
        self._preview_caption.object = "_Rendering preview…_"
        threading.Thread(
            target=self._preview_worker, args=(self._preview_gen, rows, snap),
            daemon=True, name="video-preview",
        ).start()

    def _preview_worker(self, gen, rows, snap):
        """Compose one frame at the current time step, as the export would."""
        try:
            var, level, t = snap["var_name"], snap["level"], snap["t"]
            if len(self._preview_cache) > 64:
                self._preview_cache.clear()
            units = []
            for row in rows:
                unit = []
                for label, path, is_diff, cmap, vmin, vmax in row:
                    key = (path, var, level, t, cmap, vmin, vmax,
                           snap["boundaries"])
                    img = self._preview_cache.get(key)
                    if img is None:
                        if gen != self._preview_gen:
                            return   # superseded; don't render for nothing
                        tile = _Tile(label, path, is_diff, cmap, vmin, vmax,
                                     boundaries=snap["boundaries"])
                        with _RENDER_LOCK:
                            img = tile.render(var, level, t)
                        self._preview_cache[key] = img
                    unit.append((label, img))
                units.append(unit)

            try:
                da, meta = load_e2s_field(rows[0][0][1], var, level, t)
                units_attr = da.attrs.get("units")
            except Exception:
                meta, units_attr = None, None
            title, subtitle = _header_lines(
                var, level, snap["level_dim"], units_attr, meta, t,
                snap["nframes"])

            target = snap["resolution"]
            per_row = _resolve_per_row(snap["layout"], units, target)
            frame = _compose(_arrange(units, per_row), title, subtitle)
            img, fill = _preview(frame, target)
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            how = (f"Best fit: **{per_row}** per row"
                   if LAYOUTS[snap["layout"]] is None
                   else f"**{per_row}** per row")
            caption = f"{how} — panels fill **{fill:.0%}** of the frame"
            if target is not None and fill < 0.995:
                caption += " (grey is empty space)"
            _ui(self._show_preview, gen, buf.getvalue(), caption)
        except Exception as exc:
            _ui(self._show_preview, gen, None,
                f"_Preview unavailable: {short_error(exc)}_")

    def _show_preview(self, gen, png, caption):
        if gen != self._preview_gen:
            return
        self._preview_pane.object = png
        self._preview_pane.visible = png is not None
        self._preview_caption.object = caption

    @param.depends("frames_per_second", "resolution", watch=True)
    def _refresh_summary(self):
        if self._exporting():
            return
        if self._panel_spec() is None or self.controls.time_slider.disabled:
            self._summary.object = "_Nothing rendered to export._"
            self._export_btn.disabled = True
            return
        rows = self._panel_spec(selected_only=True)
        if rows is None:
            self._summary.object = "_Select at least one simulation._"
            self._export_btn.disabled = True
            return
        n = self._nframes()
        npanels = sum(len(r) for r in rows)
        target = RESOLUTIONS[self.resolution]
        res_txt = f"{target[0]}\u00d7{target[1]}" if target else "source resolution"
        self._summary.object = (
            f"**{n}** time steps \u00d7 **{npanels}** panel(s) \u2192 "
            f"**{n / self.frames_per_second:.1f} s** of video, {res_txt}."
        )
        self._export_btn.disabled = False

    def _request_cancel(self, _event=None):
        self._cancel.set()
        self._status.object = "_Cancelling\u2026_"

    def _start(self, _event=None):
        if self._exporting():
            return
        rows = self._panel_spec(selected_only=True)
        if rows is None:
            self._status.object = (
                "**Nothing to export** \u2014 render something and select "
                "at least one simulation.")
            return
        suite_dir = self._suite_dir_fn()
        if suite_dir is None:
            self._status.object = "**Nothing to export** \u2014 no suite selected."
            return
        out_dir = _output_dir(Path(suite_dir))
        if not os.access(out_dir, os.W_OK):
            self._status.object = (
                f"**Can't save the video** \u2014 you don't have write "
                f"permission in:\n\n`{out_dir}`")
            return

        # Snapshot the controls now. The user is free to keep scrubbing the
        # live session while the export runs in the background; the video
        # reflects the state at the moment Export was pressed, not whatever
        # the widgets drift to mid-encode.
        snapshot = dict(
            var_name=self.controls.var_name,
            level=self.controls.level_value,
            level_dim=self.controls.level_dim,
            nframes=self._nframes(),
            boundaries=self._boundaries(),
            fps=self.frames_per_second,
            resolution=RESOLUTIONS[self.resolution],
            layout=self.layout,
            suite_dir=Path(suite_dir),
            out_dir=out_dir,
        )

        self._cancel.clear()
        self._export_btn.disabled = True
        self._cancel_btn.disabled = False
        for check in self._model_checks.values():
            check.disabled = True
        self._progress.value = 0
        self._progress.visible = True
        self._done.visible = False
        self._reset_open_button()
        self._status.object = "_Starting\u2026_"
        self._thread = threading.Thread(
            target=self._worker, args=(rows, snapshot),
            daemon=True, name="video-export",
        )
        self._thread.start()
        self._layout.disabled = True
        self._render_model_rows()   # now _exporting(): arrows disabled
        self.busy = True

    # -- worker ------------------------------------------------------------

    def _worker(self, rows, snap):
        # For the usage log; each early return below is a cancel.
        usage = {"outcome": "cancelled"}
        started = time.time()
        try:
            exe = _ffmpeg_exe()

            tiles = [[_Tile(label, path, is_diff, cmap, vmin, vmax,
                            boundaries=snap["boundaries"])
                      for label, path, is_diff, cmap, vmin, vmax in row]
                     for row in rows]
            flat = [t for row in tiles for t in row]

            total = snap["nframes"]
            var, level = snap["var_name"], snap["level"]

            # --- pass 1: resolve any still-unset (difference) colour ranges --
            # Field panels already carry the GUI's current range and never
            # land here; only a difference panel -- which has no single
            # GUI-visible range to copy -- needs the full sweep scanned.
            scanning = [t for t in flat if t.needs_scan]
            if scanning:
                for i in range(total):
                    if self._cancel.is_set():
                        _ui(setattr, self._status, "object", "**Export cancelled.**")
                        return
                    with _RENDER_LOCK:
                        for tile in scanning:
                            tile.scan(var, level, i)
                    _ui(setattr, self._progress, "value", int(45 * (i + 1) / total))
                    _ui(setattr, self._status, "object",
                        f"_Scanning color range {i + 1}/{total}_")
            for tile in flat:
                tile.finalize()

            render_base, render_span = (45, 45) if scanning else (0, 90)

            # --- pass 2: render frames ---------------------------------------
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            out_path = snap["out_dir"] / f"inferstudio_{var}_{stamp}.mp4"
            units = None
            # Resolved from the first frame's tile sizes, then held: every
            # tile keeps its first size (see _Tile.render), so the frame
            # does too, as libx264 requires.
            per_row = None

            with tempfile.TemporaryDirectory(prefix="inferstudio_frames_") as tmp:
                tmpdir = Path(tmp)
                for t in range(total):
                    if self._cancel.is_set():
                        _ui(setattr, self._status, "object", "**Export cancelled.**")
                        return

                    with _RENDER_LOCK:
                        composed = [
                            [(tile.label, tile.render(var, level, t))
                             for tile in row]
                            for row in tiles
                        ]
                        # The first panel's valid time and units speak for
                        # the frame, as the first panel's do on screen. The
                        # dataset is already cached by the render above.
                        try:
                            da, meta = load_e2s_field(
                                flat[0].model_dir, var, level, t)
                            units = units or da.attrs.get("units")
                        except Exception:
                            meta = None

                    if per_row is None:
                        per_row = _resolve_per_row(
                            snap["layout"], composed, snap["resolution"])
                        usage["per_row"] = per_row
                    title, subtitle = _header_lines(
                        var, level, snap["level_dim"], units, meta, t, total)
                    _compose(_arrange(composed, per_row), title,
                             subtitle).save(tmpdir / f"frame_{t:05d}.png")

                    _ui(setattr, self._progress, "value",
                        render_base + int(render_span * (t + 1) / total))
                    _ui(setattr, self._status, "object", f"_Rendered {t + 1}/{total}_")

                # --- pass 3: encode -------------------------------------------
                _ui(setattr, self._status, "object", "_Encoding with ffmpeg\u2026_")

                vf = []
                if snap["resolution"] is not None:
                    w, h = snap["resolution"]
                    # The composited frame's own aspect ratio depends on the
                    # current layout (panel count/rows) and rarely matches
                    # the target's, so scale to fit within it and letterbox
                    # the rest -- padded in the same white _BG _compose()
                    # already uses, so the bars blend with the frame rather
                    # than showing as a stark black border.
                    vf.append(
                        f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
                        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=white"
                    )
                vf.append("format=yuv420p")   # broad player compatibility

                cmd = [
                    exe, "-y",
                    "-framerate", f"{snap['fps']:g}",   # input rate = frames/s
                    "-i", str(tmpdir / "frame_%05d.png"),
                    "-vf", ",".join(vf),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-r", str(MP4_FRAME_RATE),          # output container rate
                    "-movflags", "+faststart",
                    str(out_path),
                ]
                # Polled rather than run() so Cancel also stops a long encode.
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True)
                stderr = []
                reader = threading.Thread(
                    target=lambda: stderr.extend(proc.stderr), daemon=True)
                reader.start()
                while proc.poll() is None:
                    if self._cancel.wait(0.25):
                        proc.kill()
                        proc.wait()
                        reader.join()
                        out_path.unlink(missing_ok=True)
                        _ui(setattr, self._status, "object", "**Export cancelled.**")
                        return
                reader.join()
                if proc.returncode != 0:
                    out_path.unlink(missing_ok=True)
                    tail = "".join(stderr[-12:]).strip()
                    raise RuntimeError(f"ffmpeg failed:\n```\n{tail}\n```")

            mb = out_path.stat().st_size / 1e6
            usage.update(outcome="done", mb=round(mb, 1))
            _ui(setattr, self._progress, "value", 100)
            _ui(setattr, self._status, "object", "")
            # filename first: the browser saves as soon as data arrives.
            data = base64.b64encode(out_path.read_bytes()).decode("ascii")
            _ui(setattr, self._browser_download, "filename", out_path.name)
            _ui(setattr, self._browser_download, "data", data)
            _ui(self._show_downloaded, out_path, total, mb)

        except Exception as exc:
            usage.update(outcome="error", error=short_error(exc))
            _ui(setattr, self._status, "object", f"**Export failed:** {exc}")
        finally:
            log_event(
                "video_export", suite=snap["suite_dir"].name,
                variable=snap["var_name"], level=snap["level"] or None,
                panels=[label for row in rows for label, *_ in row],
                differences=sum(is_diff for row in rows for _, _, is_diff, *_ in row),
                frames=snap["nframes"], fps=snap["fps"],
                resolution=snap["resolution"], layout=snap["layout"],
                seconds=round(time.time() - started, 1), **usage)
            _ui(self._finish)

    def _finish(self):
        self.busy = False
        self._cancel_btn.disabled = True
        for check in self._model_checks.values():
            check.disabled = False
        # The thread may not have exited yet, which would make
        # _refresh_summary think the export is still running.
        self._thread = None
        self._layout.disabled = False
        self._render_model_rows()
        self._refresh_summary()

    def _show_downloaded(self, out_path, frames, mb):
        """Tell the user the video has been downloaded, in the dialog, as a
        toast, and on the sidebar button -- the dialog may well have been
        closed while the export ran."""
        self._done.object = (
            f"### \u2705 Video downloaded\n\n"
            f"**{out_path.name}** ({frames} frames, {mb:.1f} MB) has been "
            "sent to your browser \u2014 check your Downloads folder.\n\n"
            f"A copy is also saved in `{out_path.parent}`."
        )
        self._done.visible = True
        self.open_button.name = "\u2713 Video downloaded"
        self.open_button.button_type = "success"
        if pn.state.notifications:
            pn.state.notifications.success(
                f"Video downloaded: {out_path.name} \u2014 check your "
                f"browser's Downloads folder. A copy is also saved in "
                f"{out_path.parent}.",
                duration=0,
            )

    def _reset_open_button(self):
        self.open_button.name = "Download Video"
        self.open_button.button_type = "primary"
