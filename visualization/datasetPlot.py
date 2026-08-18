# Step 1: Load datasets dynamically
from pathlib import Path
import panel as pn
import param

from dimensions import VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
from visualization.era5_plot import plot_png, NETCDF_FILE
from visualization.earth2StudioPlot import parse_variable_groups, available_levels, plot_e2s_field

pn.extension(raw_css=[Path("static/styles.css").read_text()])

# Models whose output uses the earth2studio flattened level-variable
# convention (u100, u850, ... instead of a real `level` dimension).
# TODO: WXFormer/MILES-CREDIT output is ERA5-style with a real level
# dimension and isn't handled by plot_e2s_field yet — it still needs its
# own branch here (probably routing back through era5_plot.plot_png).
EARTH2STUDIO_FORMAT_MODELS = {"AIFS", "Aurora", "Pangu"}


import matplotlib.colors as mcolors

try:
    import cmocean
    _CMOCEAN_AVAILABLE = True
except ImportError:
    _CMOCEAN_AVAILABLE = False


def _get_available_colormaps():
    """Return {name: matplotlib Colormap object} for cmocean's colormaps.
    Falls back to a small set of built-in matplotlib colormaps if cmocean
    isn't installed, so the app still works (install with `pip install
    cmocean` in the active env for the full cmocean set)."""
    if _CMOCEAN_AVAILABLE:
        names = getattr(cmocean.cm, "cmapnames", None)
        if names is None:
            # Fallback: introspect the module for Colormap instances directly
            names = [
                n for n in dir(cmocean.cm)
                if isinstance(getattr(cmocean.cm, n, None), mcolors.Colormap)
            ]
        cmaps = {}
        for name in names:
            obj = getattr(cmocean.cm, name, None)
            if isinstance(obj, mcolors.Colormap):
                cmaps[name] = obj
        if cmaps:
            return cmaps

    # cmocean not installed (or nothing found) — fall back to matplotlib builtins
    import matplotlib.pyplot as plt
    fallback_names = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]
    return {name: plt.get_cmap(name) for name in fallback_names}


def _cmap_to_hex_swatch(cmap, n=32):
    """Sample a Colormap into a list of hex colors for widget swatch preview."""
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


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

        self._time_display = pn.pane.HTML(
            pn.bind(lambda v: f"<b>Time:</b> {v}", self.time_slider.param.value),
            styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
            width=90,
            margin=0,
        )

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

        # Colormap selector — built from cmocean (or a matplotlib fallback
        # if cmocean isn't installed). self._colormaps maps name -> the
        # actual Colormap object, used to resolve the selected name back to
        # a real colormap when plotting (rather than relying on matplotlib
        # recognizing cmocean's names as globally registered strings).
        self._colormaps = _get_available_colormaps()
        swatch_options = {
            name: _cmap_to_hex_swatch(cmap) for name, cmap in self._colormaps.items()
        }
        default_cmap_name = next(iter(swatch_options), "viridis")

        if hasattr(pn.widgets, "ColorMap"):
            # NOTE: ColorMap's `value` param holds one of the option *values*
            # (the swatch color list here), not its key/name — the widget
            # separately reflects the selected key as `value_name`. So we
            # seed `value` with the actual swatch list, and link on
            # `value_name` (the string) to drive our shared `colormap` param.
            self.colormap_selector = pn.widgets.ColorMap(
                options=swatch_options,
                value=swatch_options.get(default_cmap_name),
                ncols=1,
                swatch_width=150,
                name="",
                sizing_mode="stretch_width",
            )
            self.colormap_selector.link(self, value_name="colormap")
        else:
            # Older Panel version without the ColorMap widget — fall back
            # to a plain text dropdown (no swatch preview).
            self.colormap_selector = pn.widgets.Select(
                options=list(swatch_options.keys()),
                value=default_cmap_name,
                sizing_mode="stretch_width",
            )
            self.colormap_selector.link(self, value="colormap")
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

        self._row = pn.Column(
            pn.Row(
                self._time_display,
                self.time_slider,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Variable</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.var_selector,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Level (hPa)</b>",
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
                pn.pane.HTML(
                    "<b>Colormap</b>",
                    styles={'line-height': '20px', 'font-size': '14px', 'white-space': 'nowrap'},
                    margin=0,
                ),
                self.colormap_selector,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Colormap Min</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.cmap_min_input,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            pn.Row(
                pn.pane.HTML(
                    "<b>Colormap Max</b>",
                    styles={'line-height': '30px', 'font-size': '14px', 'white-space': 'nowrap'},
                    width=90,
                    margin=0,
                ),
                self.cmap_max_input,
                align="start",
                sizing_mode="stretch_width",
                css_classes=["widget-row"],
            ),
            sizing_mode="stretch_width",
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
        leveled_vars_cf = {}
        for key in dataset_keys:
            meta = dataset_metadata.get(key) or {}
            all_vars.update(meta.get("vars2d") or [])
            all_vars.update(meta.get("vars3d") or [])
            ntime = meta.get("ntime") or 0
            if ntime > max_ntime:
                max_ntime = ntime
            for var, levels in (meta.get("leveled_vars_cf") or {}).items():
                leveled_vars_cf.setdefault(var, set()).update(levels)
        self._leveled_vars_cf = {k: sorted(v) for k, v in leveled_vars_cf.items()}

        # Time: use the longest checked dataset's range. Shorter datasets
        # get their time index clamped automatically in plot_e2s_field, so
        # scrubbing past a shorter dataset's end just holds its last step.
        end = max(max_ntime - 1, 0)
        if end <= 0:
            end = 1  # avoid Bokeh's zero-width slider error; stays disabled below regardless
        self.time_slider.end = end
        self.time_slider.disabled = not bool(dataset_keys) or max_ntime <= 0
        if self.time_index > end:
            self.time_slider.value = 0
            self.time_index = 0

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

        if levels:
            self.level_selector.options = levels
            self.level_selector.disabled = False
            if self.level_value not in levels:
                default_level = 500 if 500 in levels else levels[0]
                self.level_selector.value = default_level
                self.level_value = default_level
        else:
            self.level_selector.options = [0]
            self.level_selector.disabled = True
            self.level_selector.value = 0
            self.level_value = 0

    def panel(self):
        return self._row


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

        # Reactive view bound to the *shared* controls' time_index/var_name/
        # level_value/colormap/cmap_min/cmap_max — same pn.bind pattern
        # used for the time label in SharedPlotControls itself. Since both
        # models' plots read from these same shared params, they always
        # get the exact same color range.
        self.view = pn.bind(
            self._render,
            self.controls.param.var_name,
            self.controls.param.level_value,
            self.controls.param.time_index,
            self.controls.param.colormap,
            self.controls.param.cmap_min,
            self.controls.param.cmap_max,
        )

    def _render(self, var_name, level_value, time_index, colormap_name, cmap_min, cmap_max):
        if not var_name:
            return pn.pane.Markdown("*No variable selected*")

        cmap = self.controls._colormaps.get(colormap_name, "viridis")

        panes = []
        actual_mins = []
        actual_maxs = []
        for model in self.models:
            model_path = self.model_paths.get(model)

            if model not in EARTH2STUDIO_FORMAT_MODELS:
                panes.append(
                    pn.Column(
                        pn.pane.HTML(
                            f"<div style='text-align:center; font-size:20px; font-weight:bold; margin:0; padding:0;'>{model}</div>",
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                        pn.pane.Markdown(
                            f"*Plotting for {model} output format isn't wired up yet.*"
                        ),
                        align="center",
                        sizing_mode="stretch_width",
                        margin=0,
                    )
                )
                continue

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
                actual_mins.append(vmin_used)
                actual_maxs.append(vmax_used)
                pane = pn.pane.PNG(
                    buf,
                    sizing_mode="scale_width",
                    align="center",
                    height=None,
                    min_height=None,
                    max_height=None,
                    margin=0,
                )
            except Exception as e:
                pane = pn.pane.Markdown(f"*Error plotting {model}: {e}*")

            panes.append(
                pn.Column(
                    pn.pane.HTML(
                        f"<div style='text-align:center; font-size:20px; font-weight:bold; margin:0; padding:0;'>{model}</div>",
                        sizing_mode="stretch_width",
                        margin=0,
                    ),
                    pane,
                    align="center",
                    sizing_mode="stretch_width",
                    margin=0,
                )
            )

        # Update the Min/Max boxes to show the actual range currently in
        # effect. Only overwrite a box that's still in auto mode
        # (cmap_min/cmap_max still None) — a box the user has explicitly
        # fixed keeps showing exactly what they typed, untouched. Union
        # across models (min of mins, max of maxes) so, in auto mode, the
        # displayed range covers everything visible on screen.
        if actual_mins and actual_maxs:
            overall_min = min(actual_mins)
            overall_max = max(actual_maxs)
            if cmap_min is None:
                self.controls._set_displayed_min(float(f"{overall_min:.4g}"))
            if cmap_max is None:
                self.controls._set_displayed_max(float(f"{overall_max:.4g}"))

        if len(panes) == 1:
            return panes[0]

        return pn.Row(*panes, align="center", sizing_mode="stretch_width")

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(f"### {self.dataset}"),
            pn.panel(self.view, sizing_mode="stretch_width"),
            align="center",
            sizing_mode="stretch_width",
            height=None,
            min_height=None,
            max_height=None,
            css_classes=["plot-container"],
        )
