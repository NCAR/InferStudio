import panel as pn
import os
import param
from datetime import datetime

from tooltips import below_tooltip

SAVED_TIP = "Selected variables are saved to disk"


class VariablePicker(param.Parameterized):
    """Multi-select button group of output variables, where a variable some
    selected model doesn't write can be disabled. A disabled variable drops
    out of `value` but stays wanted, so it's back once it's enabled again.

    options: a list of variable names, or a dict of button label -> name."""

    value = param.List(default=[])

    def __init__(self, options, value=(), **params):
        super().__init__(**params)
        self._labels = (dict(options) if isinstance(options, dict)
                        else {name: name for name in options})
        self._labels_by_name = {name: label for label, name in self._labels.items()}
        self._wanted = set(value)
        self._unavailable = {}      # name -> why
        self._tips = {}             # name -> the tooltip its button shows
        self._buttons = {}
        for label, name in self._labels.items():
            button = pn.widgets.Button(name=label, button_type='primary',
                                       margin=(0, 5, 5, 0))
            button.on_click(lambda e, n=name: self._toggle(n))
            self._buttons[name] = button
        self._refresh()

    @property
    def names(self):
        return list(self._buttons)

    def set_unavailable(self, unavailable):
        """Disable the variables in unavailable, a dict of name -> the
        selected models that don't write it, and enable the rest."""
        self._unavailable = {
            name: self._disabled_tip(self._labels_by_name[name], models)
            for name, models in unavailable.items()}
        self._refresh()

    @staticmethod
    def _disabled_tip(label, models):
        who = (models[0] if len(models) == 1
               else f"{', '.join(models[:-1])} and {models[-1]}")
        verb = "doesn't" if len(models) == 1 else "don't"
        return (f"Disabled: {who} {verb} output {label}. A variable can only be "
                "saved when every selected model outputs it.")

    def _toggle(self, name):
        self._wanted ^= {name}
        self._refresh()

    def _refresh(self):
        for name, button in self._buttons.items():
            why = self._unavailable.get(name)
            button.disabled = why is not None
            if self._tips.get(name) != (why or SAVED_TIP):
                self._tips[name] = why or SAVED_TIP
                button.description = below_tooltip(self._tips[name])
            button.button_style = ('solid' if name in self._wanted and why is None
                                   else 'outline')
        self.value = [n for n in self._buttons
                      if n in self._wanted and n not in self._unavailable]

    def panel(self):
        return pn.FlexBox(*self._buttons.values(), flex_wrap='wrap',
                          sizing_mode='stretch_width')


class OutputParams:
    # Each options argument is a list of variable names, or a dict of
    # button label -> variable name.
    def __init__(self, start_path=".", width=400,
                 surface_options=('SP', 't2m'), surface_default=('SP', 't2m'),
                 ua_options=('U', 'V', 'T', 'Q'), ua_default=('U', 'V', 'T', 'Q')):
        self.current_path_val = os.path.abspath(os.path.expanduser(start_path))
        self._callback = None
        self._last_auto_name = None

        self.simulationNamePicker = pn.widgets.TextInput(
            name='Simulation Name:', 
            sizing_mode="stretch_width",
            placeholder='Enter a name for your simulation'
        )

        self.pathDisplay = pn.widgets.TextInput(
            value=self.current_path_val,
            sizing_mode="stretch_width",
        )

        self._build_picker(width)

        self.modal = pn.Modal(self.dialog, name="Select output directory", margin=0)
        self.browse_button = self.modal.create_button("toggle", name="Output Directory", button_type="primary")

        self.surfaceVars = VariablePicker(surface_options, value=surface_default)
        self.surfaceVarsGroup = pn.Row(
            pn.pane.Markdown("2D Variables"),
            self.surfaceVars.panel()
        )

        self.UAVars = VariablePicker(ua_options, value=ua_default)
        self.UAVarsGroup = pn.Row(
            pn.pane.Markdown("3D Variables"),
            self.UAVars.panel()
        )

    def set_default_simulation_name(self, models):
        """Regenerate the default simulation name from the given list of
        model names, formatted as:
            InferStudio_<ModelName1>_<ModelName2>_..._Year_Month_Day_hh:mm:ss

        Only overwrites the field if it's currently empty or still matches
        a previously auto-generated value — so if the user has typed their
        own custom name, changing the model selection afterward won't
        clobber it.
        """
        current = self.simulationNamePicker.value_input
        if current and current != self._last_auto_name:
            return  # user has typed a custom name — leave it alone

        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        model_part = "_".join(models) if models else "NoModel"
        new_name = f"InferStudio_{model_part}_{timestamp}"

        self._last_auto_name = new_name
        self.simulationNamePicker.value = new_name
        self.simulationNamePicker.value_input = new_name

    def _build_picker(self, width):
        self.currentPathDisplay = pn.widgets.TextInput(
            value=self.current_path_val,
            disabled=True,
            sizing_mode="stretch_width"
        )

        self.list_container = pn.Column(
            height=350,
            scroll=True,
            styles={"border": "1px solid #ccc", "background": "white"},
        )

        self.select_button = pn.widgets.Button(
            name="Confirm Selection ✅",
            button_type="success",
            sizing_mode="stretch_width"
        )
        self.select_button.on_click(self._select)

        self.dialog = pn.Column(
            "### 📁 Select Directory",
            self.currentPathDisplay,
            self.list_container,
            self.select_button,
            width=width,
        )

        self._refresh()

    def _refresh(self):
        self.currentPathDisplay.value = self.current_path_val

        try:
            entries = os.listdir(self.current_path_val)
            dirs = sorted(
                d for d in entries
                if os.path.isdir(os.path.join(self.current_path_val, d))
            )

            options = [".."] + dirs
            buttons = []

            for folder in options:
                btn = pn.widgets.Button(
                    name=f"📁 {folder}",
                    sizing_mode="stretch_width",
                    styles={
                        "justify-content": "flex-start",
                        "text-align": "left",
                        "display": "flex",
                        "width": "100%",
                    },
                )
                btn.on_click(lambda e, f=folder: self._navigate(f))
                buttons.append(btn)

            self.list_container.objects = buttons

        except Exception as e:
            self.list_container.objects = [
                pn.pane.Markdown(f"**Error:** {e}")
            ]

    def _navigate(self, folder):
        if folder == "..":
            new_path = os.path.dirname(self.current_path_val)
        else:
            new_path = os.path.join(self.current_path_val, folder)

        new_path = os.path.normpath(new_path)

        if os.path.isdir(new_path):
            self.current_path_val = new_path
            self._refresh()

    def _select(self, _):
        self.pathDisplay.value = self.current_path_val

        if self._callback:
            self._callback(self.current_path_val)
        self.modal.hide()

    def on_select(self, callback):
        self._callback = callback

    def panel(self):
        return pn.WidgetBox(
            "### Output Parameters",
            self.simulationNamePicker,
            pn.Row(
                self.browse_button,
                self.pathDisplay,
                self.modal,
                sizing_mode="stretch_width",
            ),
            self.surfaceVarsGroup,
            self.UAVarsGroup,
            sizing_mode="stretch_width"
        )
