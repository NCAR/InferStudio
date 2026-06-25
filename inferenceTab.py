import panel as pn
import param
import subprocess
import threading
import os
import time
import signal

from datetime import datetime, timedelta
from pathlib import Path

from outputParams import OutputParams
from timePicker import TimePicker
from commandRunner import CommandRunner

from milesCreditRunner import MilesCreditRunner
from earth2StudioRunner import Earth2StudioRunner

MILES_CREDIT_MODELS = {'WXFormer'}
EARTH2STUDIO_MODELS = {'AIFS', 'Aurora', 'FourCastNet3', 'GraphCast', 'Pangu', 'SFNO'}


class InferenceTab(param.Parameterized):
    startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))

    def __init__(self, **params):
        super().__init__(**params)

        self.modelPicker = pn.widgets.CheckButtonGroup(
            name="Select Model",
            value=['WXFormer', 'AIFS'],
            options=['WXFormer', 'AIFS', 'Aurora', 'FourCastNet3', 'GraphCast', 'Pangu', 'SFNO'],
            button_type='primary',
            button_style='outline',
            margin=(0, 5, 5, 0)
        )

        self.outputParams = OutputParams(start_path=Path.home())
        self.timePicker = TimePicker()

        self.inferenceButton = pn.widgets.Button(
            name="Run Inference",
            button_type="success",
            button_style="outline"
        )
        self.inferenceButton.on_click(self._on_run_click)

        self.cancelButton = pn.widgets.Button(
            name="Cancel",
            button_type="danger",
            disabled=True,
        )
        self.cancelButton.on_click(self._on_cancel_click)

        self.spinner = pn.indicators.LoadingSpinner(
            width=30, height=30, value=False, color="primary", visible=False
        )

        self.elapsedLabel = pn.widgets.StaticText(name="Elapsed", value="N/A")
        self.completionLabel = pn.widgets.StaticText(name="Completed at", value="N/A")

        self.commandRunner = CommandRunner()

        # Per-model state; populated fresh on each Run click
        self._processes = {}       # model -> Popen
        self._log_widgets = {}     # model -> TextAreaInput
        self._timer_running = False
        self._active_count = 0     # how many model threads are still running
        self._active_lock = threading.Lock()

        # Output log area: tabs when multiple models, plain log for one
        self.outputTabs = pn.Tabs(sizing_mode="stretch_both")
        self.statusRow = pn.Row()
        self._status_widgets = {}

    # ------------------------------------------------------------------ #
    #  Runner helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_runners(self):
        """Return list of (model_name, runner) for every selected model."""
        runners = []
        for model in self.modelPicker.value:
            if model in MILES_CREDIT_MODELS:
                runners.append((model, MilesCreditRunner()))
            elif model in EARTH2STUDIO_MODELS:
                runners.append((model, Earth2StudioRunner()))
        return runners

    def _build_config(self, model: str) -> dict:
        return {
            "simulation_name": self.outputParams.simulationNamePicker.value,
            "start_time":      self.timePicker.startDatePicker.value,
            "end_time":        self.timePicker.endDatePicker.value,
            "timestep":        self.timePicker.incrementButtons.value,
            "ua_vars":         self.outputParams.UAVars.value,
            "surface_vars":    self.outputParams.surfaceVars.value,
            "output_path":     self.outputParams.pathDisplay.value,
            "output_dir":      self.outputParams.current_path_val,
            "model":           model,
        }

    # ------------------------------------------------------------------ #
    #  UI event handlers                                                   #
    # ------------------------------------------------------------------ #

    def _on_run_click(self, event):
        runners = self._get_runners()
        if not runners:
            self._set_single_log("Error: No recognized model selected.")
            return

        # Pre-validate all runners before touching the UI
        for model, runner in runners:
            config = self._build_config(model)
            error = runner.validate(config)
            if error:
                self._set_single_log(f"[{model}] {error}")
                return

        # Build one TextAreaInput per model and populate the tab panel
        #self._log_widgets = {}
        #tabs = []
        #for model, _ in runners:
        #    widget = pn.widgets.TextAreaInput(
        #        name=model,
        #        value="",
        #        sizing_mode="stretch_both",
        #    )
        #    self._log_widgets[model] = widget
        #    tabs.append((model, widget))

        #self.outputTabs.objects = []
        #for name, widget in tabs:
        #    self.outputTabs.append((name, widget))

        self._log_widgets = {}
        self._spinners = {}

        self._status_widgets = {}
        status_items = []
        for model, _ in runners:
            pane = pn.pane.HTML(
                self._status_html(model, "running"),
                width=120,
            )
            self._status_widgets[model] = pane
            status_items.append(pane)
        self.statusRow.objects = status_items

        #def _status_html(self, model, state):
        #    # state: "running" | "done" | "error"
        #    symbol = {"running": "⟳", "done": "✓", "error": "✗"}[state]
        #    color  = {"running": "gray", "done": "green", "error": "red"}[state]
        #    return f'<div style="text-align:center;color:{color}"><b>{model}</b><br>{symbol}</div>'

        # add new method alongside other helpers, before panel():
        def _status_html(self, model: str, state: str) -> str:
            symbol = {"running": "⟳", "done": "✓", "error": "✗"}[state]
            color  = {"running": "#888888", "done": "#2ecc71", "error": "#e74c3c"}[state]
            return (
                f'<div style="text-align:center;line-height:1.4">'
                f'<span style="font-size:11px;color:#555">{model}</span><br>'
                f'<span style="font-size:20px;color:{color}">{symbol}</span>'
                f'</div>'
            )

        self.outputTabs.objects = []
        self._status_widgets = {}
        self.statusRow.objects = []
        for model, _ in runners:
            widget = pn.widgets.TextAreaInput(
                name=model,
                value="",
                sizing_mode="stretch_both",
            )
            spinner = pn.indicators.LoadingSpinner(
                width=25, height=25, value=True, color="primary", visible=True
            )
            self._log_widgets[model] = widget
            self._spinners[model] = spinner
            self.outputTabs.append((model, pn.Column(spinner, widget, sizing_mode="stretch_both")))
            pane = pn.pane.HTML(self._status_html(model, "running"), width=100)
            self._status_widgets[model] = pane
            self.statusRow.append(pane)

        self._processes = {}

        self.spinner.value = True
        self.spinner.visible = True
        self.inferenceButton.disabled = True
        self.cancelButton.disabled = False
        self.elapsedLabel.value = ""
        self.completionLabel.value = ""

        # Shared elapsed timer
        overall_start = time.time()
        self._timer_running = True

        def _tick():
            while self._timer_running:
                elapsed = time.time() - overall_start
                mins, secs = divmod(int(elapsed), 60)
                self.elapsedLabel.value = f"{mins}m {secs}s"
                time.sleep(1)

        timer_thread = threading.Thread(target=_tick, daemon=True)
        timer_thread.start()

        # Track how many model threads are still alive
        with self._active_lock:
            self._active_count = len(runners)

        def _on_model_done():
            """Called by each model thread when it finishes."""
            with self._active_lock:
                self._active_count -= 1
                all_done = self._active_count == 0
            if all_done:
                self._timer_running = False
                timer_thread.join()
                elapsed = time.time() - overall_start
                mins, secs = divmod(int(elapsed), 60)
                self.elapsedLabel.value = f"{mins}m {secs}s"
                self.completionLabel.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.spinner.value = False
                self.spinner.visible = False
                self.inferenceButton.disabled = False
                self.cancelButton.disabled = True

        for model, runner in runners:
            t = threading.Thread(
                target=self._run_model,
                args=(model, runner, _on_model_done),
                daemon=True,
            )
            t.start()

    def _on_cancel_click(self, event):
        for model, proc in list(self._processes.items()):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                self._append_log(model, "\n\nCancelled by user.")
        self.cancelButton.disabled = True

    # ------------------------------------------------------------------ #
    #  Per-model execution                                                 #
    # ------------------------------------------------------------------ #

    def _append_log(self, model: str, text: str):
        widget = self._log_widgets.get(model)
        if widget is not None:
            widget.value += text

    def _run_model(self, model: str, runner, on_done):
        """Prepare, build, and execute a single model in its own thread."""
        log = self._log_widgets[model]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        config = self._build_config(model)

        # Each model writes into <output_path>/<model_name>/
        model_output = Path(config["output_path"]) / model
        model_output.mkdir(parents=True, exist_ok=True)
        config["output_path"] = str(model_output)

        try:
            prepared = runner.prepare(config)
            config.update(prepared)
            cmd = runner.build_cmd(config)
        except Exception as e:
            self._append_log(model, f"Error during setup: {e}\n")
            on_done()
            return

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
            self._processes[model] = proc

            for line in proc.stdout:
                self._append_log(model, line)

            proc.wait()

            if proc.returncode != 0:
                self._append_log(model, f"\nExited with code {proc.returncode}\n")
            else:
                self._append_log(model, "\nDone.\n")

        except Exception as e:
            self._append_log(model, f"Error: {e}\n")
        finally:
            self._processes.pop(model, None)
            spinner = self._spinners.get(model)
            if spinner is not None:
                spinner.value = False
                spinner.visible = False
            pane = self._status_widgets.get(model)
            if pane is not None:
                returncode = self._processes.get(model)  # already popped, so check proc directly
                # proc is gone from _processes by now; use the local variable instead
                try:
                    state = "done" if proc.returncode == 0 else "error"
                except Exception:
                    state = "error"
                pane.object = self._status_html(model, state)
            on_done()
        #finally:
        #    self._processes.pop(model, None)
        #    on_done()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _set_single_log(self, message: str):
        """Show a plain error message before tabs have been built."""
        widget = pn.widgets.TextAreaInput(
            name="Log", value=message, sizing_mode="stretch_both"
        )
        self.outputTabs.objects = []
        self.outputTabs.append(("Log", widget))

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    def panel(self):
        return pn.Column(
            pn.WidgetBox(
                '# AI Model',
                self.modelPicker,
                sizing_mode='stretch_width',
            ),
            self.outputParams.panel(),
            self.timePicker.panel,
            pn.WidgetBox(
                "# Launcher",
                pn.Row(
                    self.inferenceButton, self.cancelButton,
                    self.spinner,
                    pn.Column(self.elapsedLabel, self.completionLabel)
                ),
                self.outputTabs,
                self.outputTabs,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_both',
            min_height=300
        )
