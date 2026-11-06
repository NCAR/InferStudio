# Step 1: Load datasets dynamically
import os
import threading
from functools import partial
from pathlib import Path
import panel as pn
import param
from bokeh.models import CustomJSTickFormatter

from dimensions import VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, MODEL_LEV_NAME, LAT_NAME, LON_NAME
from cf_convert import variable_long_name
from tooltips import below_tooltip
from visualization.era5_plot import plot_png, NETCDF_FILE
from visualization.earth2StudioPlot import parse_variable_groups, available_levels, plot_e2s_field
from visualization.modelDiff import compute_model_difference
from visualization.boundaries import (
    BOUNDARY_OPTIONS, DEFAULT as BOUNDARIES_DEFAULT)

pn.extension(raw_css=[Path("static/styles.css").read_text()])

# Models whose output plot_e2s_field can plot: earth2studio's layout, as
# cf_convert writes it (WXFormer's is gathered into it by
# cf_convert.wxformer_to_cf).
EARTH2STUDIO_FORMAT_MODELS = {"AIFS", "Aurora", "Pangu", "FourCastNet3", "CAMulator",
                              "DLESyMv0_ISCCP_ERA5", "GFS", "CESM2-LE", "WXFormerGen2"}


def _schedule(fn):
    """Run fn on the Bokeh document's event loop.

    Param/widget mutations made from a worker thread must not be applied
    directly: Panel needs the document lock to push the resulting change to
    the browser. pn.state.execute handles that when a server session
    exists, and falls through to a direct call in scripts and tests. See
    plotGrid._schedule, which this mirrors.
    """
    try:
        if pn.state.curdoc is not None:
            pn.state.execute(fn)
            return
    except Exception:
        pass
    fn()


from visualization import vaporColormaps

# The colormap picker in the sidebar. Bokeh's PaletteSelect draws each
# entry as a swatch with its name to the right; the open list is wider
# than the sidebar would allow with long swatches, so they're kept short
# enough for the longest name ("PurpleWhiteOrange") to fit. The CSS
# leaves the name out of the closed picker (it's shown above it instead)
# and stretches that one swatch across it; it can't reach the open list,
# which Bokeh draws in its own shadow root.
COLORMAP_SWATCH_WIDTH = 60
_SWATCH_STYLES = (
    ".bk-value.bk-entry { font-size: 0; }"
    # The closed picker's swatch is a div in an entry inside .bk-value.
    ".bk-value.bk-entry > .bk-entry { flex-grow: 1; }"
    ".bk-value .bk-entry > div { width: 100% !important; }"
)
# The list's order: the sequential maps, which suit most fields, first.
COLORMAP_ORDER = ("Sequential", "Diverging", "Highlighting")


def _swatches(reverse=False):
    """{name: hex colours} for every colormap, as the swatch widget wants
    its options; 32 colours each is plenty for a swatch."""
    suffix = vaporColormaps.REVERSED if reverse else ""
    return {name: vaporColormaps.colors(name + suffix, 32)
            for category in COLORMAP_ORDER
            for name in vaporColormaps.palettes()[category]}


def _dropdown_width(options):
    if not options:
        return 100
    longest = max(len(str(opt)) for opt in options)
    return max(80, (longest * 9) + 40)


class SharedPlotControls(param.Parameterized):
    """Owns the Variable/Level selectors shared across every plot in the
    Visualization tab. Its choices are rebuilt from the union of variables
    across whichever datasets are currently checked in the browser."""

    var_name = param.String(default="")
    level_value = param.Integer(default=0)
    # The vertical dimension level_value is on: PRES_NAME (hPa) or
    # MODEL_LEV_NAME (a model's own levels, CAMulator's).
    level_dim = param.String(default=PRES_NAME)
    time_index = param.Integer(default=0)
    colormap = param.String(default="")
    # None means "auto" (matplotlib's own per-plot scaling). Once the user
    # actually types into the Min/Max box, this becomes a real number and
    # both models' plots share that exact same fixed color range, since
    # they're both driven by these same shared params. Distinguishing
    # "user typed this" from "code just displayed the live auto value" is
    # handled via the guarded watchers below (_on_cmap_min_input etc.),
    # NOT via a simple .link() — a plain link would treat every
    # programmatic display update as a real user override too, silently
    # turning "auto" into a permanently fixed value after the very first
    # render.
    cmap_min = param.Number(default=None, allow_None=True)
    cmap_max = param.Number(default=None, allow_None=True)
    # A visualization.boundaries.BOUNDARY_OPTIONS label.
    boundaries = param.String(default=BOUNDARIES_DEFAULT)

    def __init__(self, **params):
        super().__init__(**params)

        self.level_vars = {}
        self.surface_vars = []
        # Per-variable real pressure levels captured from CF-compliant
        # files (post cf_convert.py) during scanning — see
        # scan_single_dataset's `leveled_vars_cf` in app_layout.py. Takes
        # precedence over name-parsed levels (self.level_vars) whenever a
        # variable has real levels available, since parse_variable_groups
        # can't detect them from a CF file's plain variable names (e.g.
        # "q" has no trailing digit even though it now spans 13 levels).
        self._leveled_vars_cf = {}
        # {variable: vertical dimension} for those same variables.
        self._level_dims = {}

        self.time_slider = pn.widgets.IntSlider(
            name="",
            start=0,
            end=1,
            value=0,
            disabled=True,
            show_value=False,
            sizing_mode="stretch_width",
        )
        self.time_slider.link(self, value="time_index")

        # VCR controls under the slider. Playback is a server-side periodic
        # callback rather than Panel's Player, which steps on a browser
        # timer: every step is drawn on the Bokeh server thread, so the
        # next tick can't fire until the last step's plots are done, and a
        # slow render slows playback down instead of queueing up steps.
        self._play_direction = 0        # -1 backward, 1 forward, 0 paused
        self._play_callback = None
        self._play_backward_button = self._vcr_button(
            "player-play", "Play backward", partial(self._play, -1), mirrored=True)
        self._step_back_button = self._vcr_button(
            "player-skip-back", "Step back", partial(self._step, -1))
        self._pause_button = self._vcr_button("player-pause", "Pause", self._pause)
        self._step_forward_button = self._vcr_button(
            "player-skip-forward", "Step forward", partial(self._step, 1))
        self._play_forward_button = self._vcr_button(
            "player-play", "Play forward", partial(self._play, 1))
        self._vcr_buttons = [self._play_backward_button, self._step_back_button,
                             self._pause_button, self._step_forward_button,
                             self._play_forward_button]

        # Jump straight to a step by its valid time. Options are
        # label -> step index, set in update_choices.
        self.time_select = pn.widgets.Select(
            name="",
            options={"": 0},
            value=0,
            disabled=True,
            sizing_mode="stretch_width",
        )
        self.time_slider.link(self.time_select, value="value", bidirectional=True)

        # level_selector is created before var_selector/var_name is ever
        # assigned, since _update_level_options (watched on var_name) fires
        # the instant var_name changes and would otherwise reference a
        # not-yet-created widget.
        self.level_selector = pn.widgets.Select(
            name="",
            options=[0],
            value=0,
            disabled=True,
            max_width=150,
            sizing_mode="stretch_width",
        )

        # NOTE: Select widgets reset their own value to None if the current
        # value isn't a member of `options` — and an empty options list
        # means *nothing* is a valid member, including "". That None then
        # propagates through .link() into var_name, a param.String that
        # doesn't allow None, crashing the whole render. Using [""] (a
        # non-empty list containing the empty-string placeholder) instead
        # of [] avoids ever hitting that invalid state.
        self.var_selector = pn.widgets.Select(
            name="",
            options=[""],
            value="",
            disabled=True,
            max_width=150,
            sizing_mode="stretch_width",
        )
        self.var_selector.link(self, value="var_name")
        self.level_selector.link(self, value="level_value")

        # What the selected short name means, right under the dropdown -
        # "sp" or "msl" alone means nothing to many users.
        self._var_description = pn.pane.HTML(
            pn.bind(self._describe_variable, self.var_selector.param.value),
            sizing_mode="stretch_width",
            margin=(0, 15, 5, 5),
        )

        # Colormap: VAPOR's built-in colormaps, in one list of swatches,
        # plus its Reverse Colormap. See visualization/vaporColormaps.py.
        # self._colormaps maps every name, and "<name>_r" for each
        # reversed, to its matplotlib colormap, which the plots and the
        # video export are given.
        self._colormaps = vaporColormaps.all_colormaps()
        default_cmap_name = vaporColormaps.DEFAULT
        swatches = _swatches()
        self.colormap_selector = pn.widgets.ColorMap(
            options=swatches,
            value=swatches[default_cmap_name],
            ncols=1,
            swatch_width=COLORMAP_SWATCH_WIDTH,
            swatch_height=18,
            name="",
            sizing_mode="stretch_width",
            stylesheets=[_SWATCH_STYLES],
        )
        self.colormap_reverse = pn.widgets.Checkbox(
            name="Reverse", value=False, margin=(2, 15, 0, 5), align="end")
        # The selected colormap's name, above its swatch.
        self.colormap_name = pn.pane.HTML(
            default_cmap_name,
            styles={"font-size": "13px", "color": "#4a5866"},
            margin=(4, 15, 0, 7),
        )
        self.colormap_selector.param.watch(self._on_colormap_choice, "value_name")
        self.colormap_reverse.param.watch(self._on_colormap_reverse, "value")
        self.colormap = default_cmap_name

        # Colormap Min/Max — always display the actual value currently in
        # effect (whether auto-computed from the data or user-fixed), no
        # placeholder text. Guard flags (_syncing_cmap_min/_max) let code
        # update the DISPLAYED value after each render without that write
        # being mistaken for the user manually overriding the range.
        self._syncing_cmap_min = False
        self._syncing_cmap_max = False

        self.cmap_min_input = pn.widgets.FloatInput(
            name="",
            value=0.0,
            sizing_mode="stretch_width",
        )
        self.cmap_max_input = pn.widgets.FloatInput(
            name="",
            value=0.0,
            sizing_mode="stretch_width",
        )

        def _on_cmap_min_input(event):
            if self._syncing_cmap_min:
                return  # this write came from _set_displayed_min, not the user
            self.cmap_min = event.new

        def _on_cmap_max_input(event):
            if self._syncing_cmap_max:
                return  # this write came from _set_displayed_max, not the user
            self.cmap_max = event.new

        self.cmap_min_input.param.watch(_on_cmap_min_input, 'value')
        self.cmap_max_input.param.watch(_on_cmap_max_input, 'value')

        # Natural Earth lines drawn over the maps - see PlotGrid's
        # _draw_boundaries. Label above rather than beside, like Colormap:
        # the option names are too long for the narrow half-row selects.
        self.boundaries_selector = pn.widgets.Select(
            options=list(BOUNDARY_OPTIONS),
            value=self.boundaries,
            sizing_mode="stretch_width",
            margin=(5, 15, 5, 5),
        )
        self.boundaries_selector.link(self, value="boundaries")

        # Laid out by the sidebar in two sections, Time and Variable.
        self._time_controls = pn.Column(
            self.time_slider,
            pn.FlexBox(*self._vcr_buttons, justify_content="center",
                       sizing_mode="stretch_width"),
            self.time_select,
            sizing_mode="stretch_width",
        )
        self._variable_controls = pn.Column(
            self.var_selector,
            self._var_description,
            pn.Row(
                pn.pane.HTML(
                    pn.bind(self._level_label, self.param.level_dim),
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.level_selector,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Column(
                pn.Row(
                    pn.pane.HTML(
                        "<b>Colormap</b>",
                        styles={'line-height': '20px', 'font-size': '14px', 'white-space': 'nowrap'},
                        margin=0,
                    ),
                    pn.HSpacer(),
                    self.colormap_reverse,
                    sizing_mode="stretch_width",
                ),
                self.colormap_name,
                self.colormap_selector,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self._labelled("Min", self.cmap_min_input),
                self._labelled("Max", self.cmap_max_input),
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.pane.HTML(
                    "<b>Boundaries</b>",
                    styles={'line-height': '20px', 'font-size': '14px', 'white-space': 'nowrap'},
                    margin=(0, 0, 0, 5),
                ),
                self.boundaries_selector,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    @staticmethod
    def _labelled(text, widget):
        """widget with a bold label above it, lined up with its left edge."""
        return pn.Column(
            pn.pane.HTML(
                f"<b>{text}</b>",
                styles={'line-height': '20px', 'font-size': '14px', 'white-space': 'nowrap'},
                margin=(0, 0, 0, 10),
            ),
            widget,
            sizing_mode="stretch_width",
            margin=0,
        )

    def _on_colormap_reverse(self, event):
        """Draw the swatches the other way round, keeping the selection."""
        name = self.colormap_selector.value_name
        swatches = _swatches(event.new)
        self.colormap_selector.param.update(options=swatches, value=swatches[name])
        self._on_colormap_choice()

    def _on_colormap_choice(self, *_):
        name = self.colormap_selector.value_name
        if not name:
            return
        reverse = self.colormap_reverse.value
        self.colormap_name.object = f"{name} (reversed)" if reverse else name
        self.colormap = name + (vaporColormaps.REVERSED if reverse else "")

    @staticmethod
    def _describe_variable(var):
        if not var:
            return ""
        long_name = variable_long_name(var)
        text = (f"<b>{long_name}</b>" if long_name
                else "<i>No description available</i>")
        return (
            "<div style='background:#DFEFF6; border-left:4px solid #091422;"
            " border-radius:3px; padding:6px 10px; font-size:14px;"
            " color:#091422; line-height:1.3;'>"
            f"<span style='font-family:ui-monospace,Menlo,monospace;'>{var}</span>"
            f" = {text}</div>"
        )

    def _set_displayed_min(self, value):
        """Update the Colormap Min box's displayed value without it being
        mistaken for the user manually fixing a range."""
        self._syncing_cmap_min = True
        try:
            self.cmap_min_input.value = value
        finally:
            self._syncing_cmap_min = False

    def _set_displayed_max(self, value):
        """Update the Colormap Max box's displayed value without it being
        mistaken for the user manually fixing a range."""
        self._syncing_cmap_max = True
        try:
            self.cmap_max_input.value = value
        finally:
            self._syncing_cmap_max = False

    def update_choices(self, dataset_keys, dataset_metadata):
        """Recompute variable/level/time choices from the union of
        vars2d+vars3d and the max ntime across the given dataset keys
        (typically browser.checked_items)."""
        all_vars = set()
        max_ntime = 0
        times = []
        leveled_vars_cf = {}
        level_dims = {}
        for key in dataset_keys:
            meta = dataset_metadata.get(key) or {}
            all_vars.update(meta.get("vars2d") or [])
            all_vars.update(meta.get("vars3d") or [])
            ntime = meta.get("ntime") or 0
            if ntime > max_ntime:
                max_ntime = ntime
                times = meta.get("times") or []
            for var, levels in (meta.get("leveled_vars_cf") or {}).items():
                leveled_vars_cf.setdefault(var, set()).update(levels)
            level_dims.update(meta.get("level_dims") or {})
        self._leveled_vars_cf = {k: sorted(v) for k, v in leveled_vars_cf.items()}
        self._level_dims = level_dims

        # Time: use the longest checked dataset's range. Shorter datasets
        # get their time index clamped automatically in plot_e2s_field, so
        # scrubbing past a shorter dataset's end just holds its last step.
        end = max(max_ntime - 1, 0)
        if end <= 0:
            end = 1  # avoid Bokeh's zero-width slider error; stays disabled below regardless
        self.time_slider.end = end
        # The tooltip shown while dragging gives the step's valid time
        # rather than its index.
        self.time_slider.format = CustomJSTickFormatter(
            args=dict(labels=times),
            code="return labels[Math.round(tick)] ?? tick.toFixed(0);",
        )
        self.time_slider.disabled = not bool(dataset_keys) or max_ntime <= 0
        if self.time_index > end:
            self.time_slider.value = 0
            self.time_index = 0
        # Labels must be unique to be Select options; a repeat (or a step
        # past the end of `times`) falls back to its index.
        options = {}
        for i in range(end + 1):
            label = times[i] if i < len(times) else ""
            options[label if label and label not in options else f"Step {i}"] = i
        self.time_select.options = options
        self.time_select.value = self.time_slider.value
        self.time_select.disabled = self.time_slider.disabled
        if self.time_slider.disabled:
            self._pause()
        for button in self._vcr_buttons:
            button.disabled = self.time_slider.disabled

        self.level_vars, self.surface_vars = parse_variable_groups(sorted(all_vars))

        # Variables with real CF pressure levels should be treated as
        # "leveled" (base) choices even though their name has no trailing
        # digit — remove them from surface_vars if name-parsing put them
        # there, since parse_variable_groups can't detect this from the
        # name alone.
        for var in self._leveled_vars_cf:
            if var in self.surface_vars:
                self.surface_vars.remove(var)

        base_choices = sorted(set(self.level_vars.keys()) | set(self._leveled_vars_cf.keys()))
        surface_choices = sorted(self.surface_vars)
        var_choices = base_choices + surface_choices

        self.var_selector.options = var_choices if var_choices else [""]
        self.var_selector.max_width = _dropdown_width(var_choices) if var_choices else 150
        self.var_selector.disabled = not bool(var_choices)

        if not var_choices:
            self.var_selector.value = ""
            self.var_name = ""
            self._update_level_options()
            return

        # Keep the current selection if it's still valid for the new set of
        # checked datasets; otherwise fall back to the first choice.
        if self.var_name in var_choices:
            # value unchanged -> the var_name watcher below won't fire on
            # its own, so refresh level options explicitly.
            self._update_level_options()
        else:
            self.var_selector.value = var_choices[0]  # triggers var_name -> _update_level_options

    @param.depends('var_name', watch=True)
    def _update_level_options(self):
        # Prefer real CF pressure-coordinate levels when available for
        # this variable (post cf_convert.py); fall back to the old
        # flattened-variable-name parsing for legacy (pre-conversion)
        # files, where a variable like "u500" encodes its level in the
        # name rather than a real dimension.
        cf_levels = self._leveled_vars_cf.get(self.var_name)
        if cf_levels:
            levels = [int(round(lv)) for lv in cf_levels]
        else:
            levels = available_levels(self.level_vars, self.var_name)
        # Set before the level, so whatever reacts to a level change
        # already knows which kind of level it is.
        self.level_dim = self._level_dims.get(self.var_name, PRES_NAME)

        if levels:
            self.level_selector.options = levels
            self.level_selector.disabled = False
            if self.level_value not in levels:
                # 500 hPa, or the lowest of a model's own levels (they're
                # numbered from the top).
                if self.level_dim == MODEL_LEV_NAME:
                    default_level = levels[-1]
                else:
                    default_level = 500 if 500 in levels else levels[0]
                self.level_selector.value = default_level
                self.level_value = default_level
        else:
            self.level_selector.options = [0]
            self.level_selector.disabled = True
            self.level_selector.value = 0
            self.level_value = 0

    # Milliseconds between steps while playing, when the plots draw faster
    # than that.
    PLAY_INTERVAL_MS = 500

    @staticmethod
    def _vcr_button(icon, tip, action, mirrored=False):
        """A playback button. mirrored flips its icon left to right, which
        makes Play backward out of the play triangle."""
        button = pn.widgets.Button(
            name="", icon=icon, icon_size="1.1em",
            button_type="primary", button_style="outline",
            description=below_tooltip(tip),
            width=38, margin=(0, 2, 5, 2), disabled=True,
            stylesheets=[".bk-TablerIcon { transform: scaleX(-1); }"] if mirrored else [],
        )
        button.on_click(lambda event: action())
        return button

    def _play(self, direction):
        """Step through the times in direction (1 forward, -1 backward)
        until the last (or first) one. Played from the end it's already
        at, it starts over from the other end."""
        start, end = self.time_slider.start, self.time_slider.end
        if self.time_slider.value == (end if direction > 0 else start):
            self.time_slider.value = start if direction > 0 else end
        self._play_direction = direction
        if self._play_callback is None:
            self._play_callback = pn.state.add_periodic_callback(
                self._advance, period=self.PLAY_INTERVAL_MS)
        self._show_playing()

    def _advance(self):
        start, end = self.time_slider.start, self.time_slider.end
        value = min(max(self.time_slider.value + self._play_direction, start), end)
        self.time_slider.value = value
        if value in (start, end):
            self._pause()

    def _pause(self):
        if self._play_callback is not None:
            self._play_callback.stop()
            self._play_callback = None
        self._play_direction = 0
        self._show_playing()

    def _step(self, direction):
        self._pause()
        self.time_slider.value = min(max(self.time_slider.value + direction,
                                         self.time_slider.start), self.time_slider.end)

    def _show_playing(self):
        """Fill in the play button for the direction playing, if any."""
        for button, direction in ((self._play_backward_button, -1),
                                  (self._play_forward_button, 1)):
            button.button_style = "solid" if self._play_direction == direction else "outline"

    @staticmethod
    def _level_label(level_dim):
        text = "Model level" if level_dim == MODEL_LEV_NAME else "Level (hPa)"
        return f"<b>{text}</b>"

    def time_panel(self):
        """The time slider, playback buttons and timestamp dropdown."""
        return self._time_controls

    @staticmethod
    def _step_count_html(end, disabled):
        # A disabled slider's end is a placeholder (see update_choices).
        count = 0 if disabled else end + 1
        return f"NumTS: {count}"

    def time_count(self):
        """How many time steps the slider spans, for the Time section's
        header."""
        return pn.pane.HTML(
            pn.bind(self._step_count_html, self.time_slider.param.end,
                    self.time_slider.param.disabled),
            styles={'font-size': '13px', 'color': '#555', 'white-space': 'nowrap'},
            margin=(5, 8, 0, 5),
        )

    def variable_panel(self):
        """Variable, level, colormap and boundaries."""
        return self._variable_controls

    def _mirror_select(self, src):
        """A second Select kept in step with `src`, both ways.

        A Panel widget can sit in only one layout, so the Statistics tab
        gets its own copies of the Variable and Level selectors. Options,
        enabled state and width follow `src`; picking a value in either
        one sets the other. The guard stops `dst` writing back while it is
        being updated from `src` - assigning new options makes a Select
        reset its own value, which would otherwise be pushed back.
        """
        dst = pn.widgets.Select(
            name="", options=src.options, value=src.value,
            disabled=src.disabled, max_width=src.max_width,
            sizing_mode="stretch_width")
        syncing = {"on": False}

        def _from_src(*events):
            syncing["on"] = True
            try:
                for ev in events:
                    if getattr(dst, ev.name) != ev.new:
                        setattr(dst, ev.name, ev.new)
                if dst.value != src.value:
                    dst.value = src.value
            finally:
                syncing["on"] = False

        def _from_dst(ev):
            if not syncing["on"] and src.value != ev.new:
                src.value = ev.new

        src.param.watch(_from_src, ["options", "value", "disabled",
                                    "max_width"])
        dst.param.watch(_from_dst, "value")
        return dst

    def stats_panel(self):
        """The Variable and Level controls for the Statistics tab's
        Variable section, mirroring the Visualization tab's (see
        _mirror_select)."""
        def label(text):
            # text is a string, or a bound function that returns the HTML.
            return pn.pane.HTML(
                f"<b>{text}</b>" if isinstance(text, str) else text,
                styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                width=90,
                margin=0,
            )

        var_selector = self._mirror_select(self.var_selector)
        level_selector = self._mirror_select(self.level_selector)
        return pn.Column(
            var_selector,
            pn.pane.HTML(
                pn.bind(self._describe_variable, self.var_selector.param.value),
                sizing_mode="stretch_width",
                margin=(0, 15, 5, 5),
            ),
            pn.Row(label(pn.bind(self._level_label, self.param.level_dim)),
                   level_selector, align="start",
                   sizing_mode="stretch_width", css_classes=["widget-row"]),
            sizing_mode="stretch_width",
        )


class DatasetPlot2(param.Parameterized):
    dataset = param.String()
    metadata = param.Dict(default={})

    def __init__(self, controls, **params):
        super().__init__(**params)

        self.controls = controls
        self.metadata = self.metadata[self.dataset]

        # Suite entries (from scan_simulation_suite) carry a "models" dict
        # with each model's own path; older single-directory entries don't.
        if "models" in self.metadata:
            self.models = sorted(self.metadata["models"].keys())
            self.model_paths = {m: self.metadata["models"][m]["path"] for m in self.models}
        else:
            self.models = [self.dataset]
            self.model_paths = {self.dataset: self.metadata["path"]}

        # Per-model reactive state. Built once here (not rebuilt on every
        # variable/level/time change) so a model's "Compute Difference"
        # dropdown selection survives ordinary control changes rather
        # than resetting back to "no diff" on every render.
        self._diff_selectors = {}       # model -> Select widget
        self._diff_slots = {}           # model -> Column that holds the diff card (empty when none selected)
        # model -> difference file, for whichever diffs are
        # currently displayed. compute_model_difference already returns
        # this path; retaining it lets export_panels() below describe the
        # diff cards without recomputing anything.
        self._diff_paths = {}

        self._sections = pn.Column(sizing_mode="stretch_width")
        for model in self.models:
            self._sections.append(self._build_model_section(model))

    def _model_label(self, text):
        return pn.pane.HTML(
            f"<div style='text-align:center; font-size:20px; font-weight:bold; margin:0; padding:0;'>{text}</div>",
            sizing_mode="stretch_width",
            margin=0,
        )

    def _build_model_section(self, model):
        if model not in EARTH2STUDIO_FORMAT_MODELS:
            return pn.Column(
                self._model_label(model),
                pn.pane.Markdown(f"*Plotting for {model} output format isn't wired up yet.*"),
                align="center",
                sizing_mode="stretch_width",
                margin=0,
                css_classes=["plot-container"],
            )

        # The model's own reactive plot, bound to the shared controls —
        # same pn.bind pattern used for the time label in SharedPlotControls.
        model_view = pn.bind(
            self._render_model,
            model,
            self.controls.param.var_name,
            self.controls.param.level_value,
            self.controls.param.time_index,
            self.controls.param.colormap,
            self.controls.param.cmap_min,
            self.controls.param.cmap_max,
        )

        other_models = [m for m in self.models if m != model and m in EARTH2STUDIO_FORMAT_MODELS]
        diff_options = ["\u2014"] + [f"{model} minus {other}" for other in other_models]
        diff_selector = pn.widgets.Select(
            name="Compute Difference",
            options=diff_options,
            value="\u2014",
            width=220,
        )
        self._diff_selectors[model] = diff_selector

        diff_slot = pn.Column(
            sizing_mode="stretch_width",
            styles={"min-width": "0"},
        )  # empty until a diff is picked
        self._diff_slots[model] = diff_slot

        def _on_diff_change(event, model=model):
            self._on_diff_selected(model, event.new)

        diff_selector.param.watch(_on_diff_change, 'value')

        model_card = pn.Column(
            self._model_label(model),
            pn.panel(model_view, sizing_mode="stretch_width"),
            pn.Row(
                pn.Spacer(sizing_mode="stretch_width"),
                diff_selector,
                sizing_mode="stretch_width",
            ),
            align="center",
            sizing_mode="stretch_width",
            margin=0,
            css_classes=["plot-container"],
            styles={"min-width": "0"},
        )

        return pn.Row(
            model_card,
            diff_slot,
            sizing_mode="stretch_width",
            styles={
                "display": "grid",
                "grid-template-columns": "1fr 1fr",
                "gap": "10px",
                "width": "100%",
            },
        )

    def _render_model(self, model, var_name, level_value, time_index, colormap_name, cmap_min, cmap_max):
        if not var_name:
            return pn.pane.Markdown("*No variable selected*")

        cmap = self.controls._colormaps.get(colormap_name, "viridis")
        model_path = self.model_paths.get(model)

        try:
            buf, vmin_used, vmax_used = plot_e2s_field(
                model_dir=model_path,
                base_or_var=var_name,
                level=level_value,
                t=time_index,
                cmap=cmap,
                vmin=cmap_min,
                vmax=cmap_max,
            )
            # Reflect the range actually used back into the Min/Max boxes.
            # When cmap_min/cmap_max are None, plot_e2s_field auto-scaled
            # from this variable's own data and vmin_used/vmax_used are the
            # only place that range exists - without this the boxes just
            # sit at their initial 0.0 forever, regardless of the variable
            # plotted. Guarded via _set_displayed_min/max so this write
            # isn't mistaken for the user fixing an explicit range.
            self.controls._set_displayed_min(vmin_used)
            self.controls._set_displayed_max(vmax_used)
            return pn.pane.PNG(
                buf,
                sizing_mode="scale_width",
                align="center",
                height=None,
                min_height=None,
                max_height=None,
                margin=0,
            )
        except Exception as e:
            return pn.pane.Markdown(f"*Error plotting {model}: {e}*")

    def _on_diff_selected(self, model, selected_option):
        slot = self._diff_slots[model]

        if selected_option == "\u2014":
            slot.objects = []
            self._diff_paths.pop(model, None)   # no diff card on screen anymore
            return

        # "AIFS minus Aurora" -> other = "Aurora"
        other = selected_option.split(" minus ", 1)[1]

        selector = self._diff_selectors[model]
        selector.disabled = True
        slot.objects = [
            pn.Column(
                pn.layout.VSpacer(),
                pn.indicators.LoadingSpinner(
                    value=True, width=120, height=120, align="center"),
                pn.pane.Markdown(
                    "**Computing difference \u2014 a full suite can take a "
                    "minute.**",
                    align="center",
                ),
                pn.layout.VSpacer(),
                align="center",
                sizing_mode="stretch_width",
                min_height=420,
            )
        ]

        # Off the Bokeh callback thread: compute_model_difference over a
        # full suite can take minutes, and running it inline here would
        # block this callback's return - which is also when Panel flushes
        # pending changes to the browser. The spinner set above would never
        # reach the client; the UI would just freeze with no sign anything
        # was happening until the diff card popped in at the very end. See
        # plotGrid.py's _ensure_diff_async, which has the same shape for
        # the same reason.
        def work():
            try:
                diff_path = compute_model_difference(
                    self.model_paths[model], self.model_paths[other],
                    model, other,
                )
            except Exception as e:
                _schedule(partial(self._finish_diff_error, model,
                                  selected_option, e))
                return
            _schedule(partial(self._finish_diff_ready, model,
                              selected_option, other, diff_path))

        threading.Thread(
            target=work, daemon=True,
            name=f"diff-{model}-minus-{other}").start()

    def _stale_diff(self, model, selected_option):
        """True if the user picked a different pair while this one ran."""
        return self._diff_selectors[model].value != selected_option

    def _finish_diff_error(self, model, selected_option, exc):
        self._diff_selectors[model].disabled = False
        if self._stale_diff(model, selected_option):
            return   # a newer selection is already showing/computing
        self._diff_paths.pop(model, None)   # nothing renderable to export
        self._diff_slots[model].objects = [
            pn.pane.Markdown(f"*Error computing difference: {exc}*")
        ]

    def _finish_diff_ready(self, model, selected_option, other, diff_path):
        self._diff_selectors[model].disabled = False
        if self._stale_diff(model, selected_option):
            return   # a newer selection is already showing/computing

        self._diff_paths[model] = diff_path

        label = f"{model} minus {other}"
        diff_view = pn.bind(
            self._render_diff,
            diff_path,
            self.controls.param.var_name,
            self.controls.param.level_value,
            self.controls.param.time_index,
        )
        diff_card = pn.Column(
            self._model_label(label),
            pn.panel(diff_view, sizing_mode="stretch_width"),
            align="center",
            sizing_mode="stretch_width",
            margin=0,
            css_classes=["plot-container"],
        )
        self._diff_slots[model].objects = [diff_card]

    def _render_diff(self, diff_path, var_name, level_value, time_index):
        if not var_name:
            return pn.pane.Markdown("*No variable selected*")

        try:
            # First pass: auto-range, purely to discover the actual
            # min/max of this difference field.
            _, vmin_auto, vmax_auto = plot_e2s_field(
                model_dir=diff_path,
                base_or_var=var_name,
                level=level_value,
                t=time_index,
                cmap=vaporColormaps.colormap(vaporColormaps.DIFF),
                vmin=None,
                vmax=None,
            )
            # Second pass: symmetric range around zero, so the diverging
            # colormap's center (white/neutral) actually lands on zero
            # difference rather than on some arbitrary asymmetric midpoint.
            max_abs = max(abs(vmin_auto), abs(vmax_auto))
            buf, _, _ = plot_e2s_field(
                model_dir=diff_path,
                base_or_var=var_name,
                level=level_value,
                t=time_index,
                cmap=vaporColormaps.colormap(vaporColormaps.DIFF),
                vmin=-max_abs,
                vmax=max_abs,
            )
            return pn.pane.PNG(
                buf,
                sizing_mode="scale_width",
                align="center",
                height=None,
                min_height=None,
                max_height=None,
                margin=0,
            )
        except Exception as e:
            return pn.pane.Markdown(f"*Error plotting difference: {e}*")

    def export_panels(self):
        """Describe what this plot grid is currently rendering.

        Returns a list of rows, each a list of (label, model_dir, is_diff)
        tuples, mirroring the on-screen layout: one row per plottable
        model, with its difference card alongside when one is selected.
        Models without a wired-up plotting branch are omitted, since they
        have nothing to render into a frame.

        Deliberately returns plain tuples rather than any videoExport type,
        so this module stays independent of the export machinery.
        """
        rows = []
        for model in self.models:
            if model not in EARTH2STUDIO_FORMAT_MODELS:
                continue
            row = [(model, self.model_paths[model], False)]
            diff_path = self._diff_paths.get(model)
            if diff_path:
                row.append((self._diff_selectors[model].value, diff_path, True))
            rows.append(row)
        return rows

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(f"### {self.dataset}"),
            self._sections,
            align="center",
            sizing_mode="stretch_width",
        )
