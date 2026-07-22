# Step 1: Load datasets dynamically
from pathlib import Path
import panel as pn
import param

from era5_plot import plot_png, NETCDF_FILE, VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
from earth2StudioPlot import parse_variable_groups, available_levels, plot_e2s_field

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

    def __init__(self, **params):
        super().__init__(**params)

        self.level_vars = {}
        self.surface_vars = []

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
            sizing_mode="stretch_width",
        )

    def update_choices(self, dataset_keys, dataset_metadata):
        """Recompute variable/level/time choices from the union of
        vars2d+vars3d and the max ntime across the given dataset keys
        (typically browser.checked_items)."""
        all_vars = set()
        max_ntime = 0
        for key in dataset_keys:
            meta = dataset_metadata.get(key) or {}
            all_vars.update(meta.get("vars2d") or [])
            all_vars.update(meta.get("vars3d") or [])
            ntime = meta.get("ntime") or 0
            if ntime > max_ntime:
                max_ntime = ntime

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

        base_choices = sorted(self.level_vars.keys())
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
        levels = available_levels(self.level_vars, self.var_name)
        if levels:
            self.level_selector.options = levels
            self.level_selector.disabled = False
            if self.level_value not in levels:
                self.level_selector.value = levels[0]
                self.level_value = levels[0]
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
        # level_value — same pn.bind pattern used for the time label in
        # SharedPlotControls itself.
        self.view = pn.bind(
            self._render,
            self.controls.param.var_name,
            self.controls.param.level_value,
            self.controls.param.time_index,
            self.controls.param.colormap,
        )

    def _render(self, var_name, level_value, time_index, colormap_name):
        if not var_name:
            return pn.pane.Markdown("*No variable selected*")

        cmap = self.controls._colormaps.get(colormap_name, "viridis")

        panes = []
        for model in self.models:
            model_path = self.model_paths.get(model)

            if model not in EARTH2STUDIO_FORMAT_MODELS:
                panes.append(
                    pn.Column(
                        pn.pane.Markdown(
                            f"**{model}**",
                            styles={"text-align": "center", "font-size": "20px", "width": "100%"},
                            sizing_mode="stretch_width",
                            margin=(0, 0, 4, 0),
                        ),
                        pn.pane.Markdown(
                            f"*Plotting for {model} output format isn't wired up yet.*"
                        ),
                        align="center",
                        sizing_mode="stretch_width",
                    )
                )
                continue

            try:
                buf = plot_e2s_field(
                    model_dir=model_path,
                    base_or_var=var_name,
                    level=level_value,
                    t=time_index,
                    cmap=cmap,
                )
                pane = pn.pane.PNG(
                    buf,
                    sizing_mode="stretch_width",
                    align="center",
                    height=None,
                    min_height=None,
                    max_height=None,
                )
            except Exception as e:
                pane = pn.pane.Markdown(f"*Error plotting {model}: {e}*")

            panes.append(
                pn.Column(
                    pn.pane.Markdown(
                        f"**{model}**",
                        styles={"text-align": "center", "font-size": "20px", "width": "100%"},
                        sizing_mode="stretch_width",
                        margin=(0, 0, 4, 0),
                    ),
                    pane,
                    align="center",
                    sizing_mode="stretch_width",
                )
            )

        if len(panes) == 1:
            return panes[0]

        return pn.Row(*panes, align="center", sizing_mode="stretch_width")

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(f"### {self.dataset}"),
            self.view,
            align="center",
            sizing_mode="stretch_width",
            height=None,
            min_height=None,
            max_height=None,
            css_classes=["plot-container"],
        )
