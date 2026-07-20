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


class DatasetPlot2(param.Parameterized):
    dataset = param.String()
    var_name = param.String(default="")
    level_value = param.Integer(default=0)
    time_index = param.Integer()
    metadata = param.Dict(default={})

    def __init__(self, **params):
        super().__init__(**params)

        self.metadata = self.metadata[self.dataset]

        # Suite entries (from scan_simulation_suite) carry a "models" dict
        # with each model's own path; older single-directory entries don't.
        if "models" in self.metadata:
            self.models = sorted(self.metadata["models"].keys())
            self.model_paths = {m: self.metadata["models"][m]["path"] for m in self.models}
        else:
            self.models = [self.dataset]
            self.model_paths = {self.dataset: self.metadata["path"]}

        all_vars = (self.metadata.get("vars2d") or []) + (self.metadata.get("vars3d") or [])
        self.level_vars, self.surface_vars = parse_variable_groups(all_vars)

        base_choices = sorted(self.level_vars.keys())
        surface_choices = sorted(self.surface_vars)
        var_choices = base_choices + surface_choices

        def get_dropdown_width(options):
            if not options:
                return 100
            longest = max(len(str(opt)) for opt in options)
            return max(80, (longest * 9) + 40)

        # IMPORTANT: level_selector must exist *before* self.var_name is
        # ever assigned. _update_level_options is watched on 'var_name' and
        # fires the instant var_name is set (including this very first
        # assignment below) — if level_selector doesn't exist yet at that
        # point, the watcher raises AttributeError.
        self.level_selector = pn.widgets.Select(
            name="",
            options=[0],
            value=0,
            disabled=True,
            max_width=100,
            sizing_mode="stretch_width",
        )

        self.var_name = var_choices[0] if var_choices else ""

        self.var_selector = pn.widgets.Select(
            name="",
            options=var_choices,
            value=self.var_name,
            max_width=get_dropdown_width(var_choices),
            sizing_mode="stretch_width",
        )
        self.var_selector.link(self, value="var_name")
        self.level_selector.link(self, value="level_value")

        self.var_row = pn.Row(
            pn.widgets.StaticText(value="<b>Variable</b>", width=70, align="center"),
            self.var_selector,
            pn.widgets.StaticText(value="<b>Level (hPa)</b>", width=90, align="center"),
            self.level_selector,
            align="center",
            sizing_mode="stretch_width",
            max_width=800,
            css_classes=["widget-row"],
        )

        self.time_slider = pn.widgets.IntSlider(
            name="",
            start=0,
            end=max(self.metadata.get("ntime", 1) - 1, 0),
            value=self.time_index,
            show_value=False,
            sizing_mode="stretch_width",
        )
        self.time_slider.link(self, value="time_index")

        time_display = pn.pane.HTML(
            pn.bind(lambda v: f"<b>Time:</b> {v}", self.time_slider.param.value),
            styles={'line-height': '30px', 'font-size': '14px'},
            width=50,
            margin=0,
        )

        self.slider_row = pn.Row(
            time_display,
            self.time_slider,
            align="center",
            sizing_mode="stretch_width",
            max_width=800,
            css_classes=["widget-row"],
        )

    @param.depends('var_name', watch=True)
    def _update_level_options(self):
        levels = available_levels(self.level_vars, self.var_name)
        if levels:
            self.level_selector.options = levels
            self.level_selector.disabled = False
            self.level_selector.value = levels[0]
            self.level_value = levels[0]
        else:
            # Surface variable (t2m, msl, sp, ...) — no level to pick
            self.level_selector.options = [0]
            self.level_selector.disabled = True
            self.level_selector.value = 0
            self.level_value = 0

    @pn.depends("time_index", "level_value", "var_name")
    def view(self):
        panes = []
        for model in self.models:
            model_path = self.model_paths.get(model)

            if model not in EARTH2STUDIO_FORMAT_MODELS:
                panes.append(
                    pn.Column(
                        pn.pane.Markdown(f"**{model}**", margin=(0, 0, 0, 0)),
                        pn.pane.Markdown(
                            f"*Plotting for {model} output format isn't wired up yet.*"
                        ),
                        align="center",
                    )
                )
                continue

            try:
                buf = plot_e2s_field(
                    model_dir=model_path,
                    base_or_var=self.var_name,
                    level=self.level_value,
                    t=self.time_index,
                )
                pane = pn.pane.PNG(
                    buf,
                    sizing_mode="scale_width",
                    align="center",
                    height=None,
                    min_height=None,
                    max_height=None,
                )
            except Exception as e:
                pane = pn.pane.Markdown(f"*Error plotting {model}: {e}*")

            panes.append(
                pn.Column(
                    pn.pane.Markdown(f"**{model}**", margin=(0, 0, 0, 0)),
                    pane,
                    align="center",
                )
            )

        if len(panes) == 1:
            return panes[0]

        return pn.GridBox(*panes, ncols=min(len(panes), 2), align="center")

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(f"### {self.dataset}"),
            self.var_row,
            self.slider_row,
            self.view,
            align="center",
            sizing_mode="stretch_width",
            height=None,
            min_height=None,
            max_height=None,
            css_classes=["plot-container"],
        )
