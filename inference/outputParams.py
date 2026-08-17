import panel as pn
import os
import param
from datetime import datetime

class OutputParams:
    def __init__(self, start_path=".", width=400):
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

        self.surfaceVars = pn.widgets.CheckButtonGroup(
            name="Surface Variables",
            #value=['SP'],
            value=['SP','t2m','V500','U500','T500','Z500','Q500'],
            options=['SP','t2m','V500','U500','T500','Z500','Q500'],
            button_type='primary',
            button_style='outline',
            #disabled=True
        )
        self.surfaceVarsGroup = pn.Row(
            pn.pane.Markdown("Surface Variables"),
            self.surfaceVars
        )

        self.UAVars = pn.widgets.CheckButtonGroup(
            name="Upper Air Variables",
            #value=['U'],
            value=['U','V','T','Q'],
            options=['U','V','T','Q'],
            button_type='primary',
            button_style='outline',
            #disabled=True
        )
        self.UAVarsGroup = pn.Row(
            pn.pane.Markdown("Upper Air Variables"),
            self.UAVars
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
            "# Output Parameters",
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
