import panel as pn
import param
import threading
import os
import time
import traceback

from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

from inference.outputParams import OutputParams
from dimensions import model_file_stem
from tooltips import below_tooltip
from tabStyles import TAB_HEADER_CSS
from usage_log import log_event, short_error
from inference.timePicker import TimePicker
from inference.climateTimePicker import ClimateTimePicker
from inference.commandRunner import CommandRunner

from inference.milesCreditRunner import MilesCreditRunner, MilesCreditGen2Runner
from inference.earth2StudioRunner import Earth2StudioRunner
from inference.pbsJob import (SYSTEMS, LogTail, ModelRun, PBSJob, estimate_walltime,
                              format_duration, model_seconds, project_account)

# WXFormer runs on MILES-CREDIT's gen1 pipeline, WXFormerGen2 on gen2 (see
# MilesCreditGen2Runner).
MILES_CREDIT_MODEL_LIST = ['WXFormer', 'WXFormerGen2']
MILES_CREDIT_MODELS = frozenset(MILES_CREDIT_MODEL_LIST)
EARTH2STUDIO_MODEL_LIST = ['AIFS', 'Aurora', 'Pangu', 'FourCastNet3', 'CAMulator']
EARTH2STUDIO_MODELS = frozenset(EARTH2STUDIO_MODEL_LIST)
CLIMATE_MODEL_LIST = ['CAMulator']
WEATHER_MODEL_LIST = [m for m in MILES_CREDIT_MODEL_LIST + EARTH2STUDIO_MODEL_LIST
                      if m not in CLIMATE_MODEL_LIST]
#EARTH2STUDIO_MODELS = {'AIFS', 'Aurora', 'Pangu', 'FourCastNet3', 'GraphCast', 'SFNO'}

# Runs in the browser whenever a log box's text changes: keeps it scrolled
# to the newest output, unless the user has scrolled up to read earlier
# lines - then it stays put until they scroll back to the bottom.
LOG_AUTOSCROLL_JS = """
const view = Bokeh.index.find_one(cb_obj);
const el = view == null ? null : view.input_el;
if (el == null) { return; }
if (!el._autoscroll) {
  el._autoscroll = true;
  el._stick = true;
  el.addEventListener("scroll", () => {
    el._stick = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
  });
}
if (el._stick) {
  requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
}
"""

# Button labels for models whose name (also their output directory) isn't
# what the button should say.
MODEL_LABELS = {
    'WXFormer': 'WXFormer (gen 1)',
    'WXFormerGen2': 'WXFormer (gen 2)',
}

# Short, hover-friendly descriptions shown as a tooltip on each model button.
MODEL_DESCRIPTIONS = {
    'WXFormer': (
        "NCAR MILES-CREDIT's own transformer-based weather model (1 hour, "
        "0.25 degree), run through MILES-CREDIT's original gen1 pipeline. "
        "Writes ERA5-style output with a real vertical-level dimension."
    ),
    'WXFormerGen2': (
        "The same WXFormer weights run through MILES-CREDIT's gen2 realtime "
        "pipeline (rollout_realtime_gen2), with a config converted by "
        "`credit convert`. Needs miles-credit 2026.2 or later (the InferStudio "
        "conda environment). Saves every hourly step."
    ),
    'AIFS': (
        "ECMWF's AI Forecasting System, run through Earth2Studio (PyTorch). "
        "Requires a source-compiled flash_attn."
    ),
    'Aurora': (
        "Microsoft's Aurora foundation model for weather forecasting, run "
        "through Earth2Studio (PyTorch)."
    ),
    'Pangu': (
        "Huawei's Pangu-Weather model, run through Earth2Studio on "
        "GPU-enabled ONNX Runtime. Fast, and a reasonable baseline."
    ),
    'FourCastNet3': (
        "NVIDIA's FourCastNet 3, a spherical Fourier Neural Operator model, "
        "run through Earth2Studio (PyTorch + makani)."
    ),
    'CAMulator': (
        "NSF NCAR's emulator of the CAM6 climate model (1 degree, 6-hourly), "
        "run through Earth2Studio (PyTorch). Output is on its 32 model "
        "levels, 1 at the model top; Q is total water. Starts only "
        "from its shipped CAM states: 00Z on 1 January or 1 July, 1980-2014."
    ),
}


class ModelPicker(param.Parameterized):
    """Multi-select button group where each button shows a hover tooltip
    describing the model it selects."""

    value = param.List(default=[])

    def __init__(self, options, descriptions=None, labels=None, **params):
        super().__init__(**params)
        descriptions = descriptions or {}
        labels = labels or {}
        self._buttons = {}
        for name in options:
            button = pn.widgets.Button(
                name=labels.get(name, name),
                button_type='primary',
                button_style='solid' if name in self.value else 'outline',
                description=(below_tooltip(descriptions[name])
                             if name in descriptions else None),
                margin=(0, 5, 5, 0),
            )
            button.on_click(partial(self._on_click, name))
            self._buttons[name] = button
        # Wraps onto a second line rather than spilling out of a narrow box.
        self.row = pn.FlexBox(*self._buttons.values(), flex_wrap='wrap',
                              sizing_mode='stretch_width', margin=(0, 5, 5, 0))

    def _on_click(self, name, event):
        selected = set(self.value)
        if name in selected:
            selected.discard(name)
            self._buttons[name].button_style = 'outline'
        else:
            selected.add(name)
            self._buttons[name].button_style = 'solid'
        # Preserve the original option ordering rather than click order.
        self.value = [m for m in self._buttons if m in selected]

    def panel(self):
        return self.row


# Outline for each box in the settings grid; the accent is the app's
# primary blue, as on the selected tab.
SETTINGS_BOX_STYLES = {
    'border': '1px solid #cfd6de',
    'border-top': '4px solid #007bff',
    'border-radius': '6px',
}


# PBS job states a job can wait in, as shown on the Launcher.
_STATE_NAMES = {'Q': 'queued', 'H': 'held', 'W': 'waiting'}


@dataclass
class _Run:
    """One press of Run Inference: the GPUs asked for, the PBS job once it's
    submitted, and when it was submitted, started and ended (Unix seconds).
    Each run has its own, so a cancelled run's thread winding down can't
    touch the next run's state."""
    system: str
    gpu_type: str
    n_gpus: int
    job: PBSJob = None
    submitted_at: float = None
    started_at: float = None
    ended_at: float = None
    tick: object = None
    cancel: threading.Event = field(default_factory=threading.Event)


class ModelTab(param.Parameterized):
    """Picks models, output and time settings, then runs the selected models
    one after another. Subclasses choose the models and the settings."""

    outputDirectory = param.String(default="")

    MODELS = []
    DEFAULT_MODELS = []

    def __init__(self, **params):
        super().__init__(**params)

        self.modelPicker = ModelPicker(
            value=list(self.DEFAULT_MODELS),
            options=self.MODELS,
            descriptions=MODEL_DESCRIPTIONS,
            labels=MODEL_LABELS,
        )

        self.outputParams = self._make_output_params(
            Path(f"/glade/derecho/scratch/{os.environ['USER']}"))
        self.outputParams.simulationNamePicker.param.watch(
            lambda e: self._clear_highlight(self.outputParams.simulationNamePicker),
            'value_input'
        )
        self.modelPicker.param.watch(self._update_default_sim_name, 'value')
        self._update_default_sim_name()

        self.timePicker = self._make_time_picker()

        # --- Compute: where the PBS job runs --- #
        self.systemPicker = pn.widgets.RadioButtonGroup(
            name="System", options=list(SYSTEMS), value='Derecho',
            button_type='primary', button_style='outline',
        )
        # Unnamed: the Compute Settings layout labels these three itself, so
        # the labels and the widgets each line up on one row.
        self.gpuTypePicker = pn.widgets.Select(width=220)
        self.gpuCountPicker = pn.widgets.Select(width=80)
        self.walltimeLabel = pn.widgets.StaticText(name="Wall time (estimated)", value="")
        self._account = project_account()
        self.accountLabel = pn.widgets.StaticText(
            name="Project account",
            value=self._account or "unknown - start InferStudio from NCAR Open OnDemand",
        )
        self.systemPicker.param.watch(self._update_gpu_options, 'value')
        self.modelPicker.param.watch(self._update_gpu_options, 'value')
        self.gpuCountPicker.param.watch(self._update_walltime, 'value')
        self.modelPicker.param.watch(self._update_walltime, 'value')
        self.timePicker.param.watch(self._update_walltime, 'end_date')
        self._update_gpu_options()

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

        self.jobLabel = pn.widgets.StaticText(name="PBS job", value="N/A")
        self.queueLabel = pn.widgets.StaticText(name="Time in queue", value="N/A")
        self.elapsedLabel = pn.widgets.StaticText(name="Elapsed", value="N/A")
        self.completionLabel = pn.widgets.StaticText(name="Completed at", value="N/A")
        self.completionPathLabel = pn.widgets.StaticText(name="", value="")

        self.commandRunner = CommandRunner()

        # Per-model state; populated fresh on each Run click
        self._log_widgets = {}      # model -> TextAreaInput
        self._spinners = {}         # model -> LoadingSpinner
        self._status_widgets = {}   # model -> HTML pane
        self._failed_models = []    # models that errored out this run
        self._run = None            # the current _Run
        self._periodic_cb = None
        self._doc = None

        self.outputTabs = pn.Tabs(sizing_mode="stretch_width", height=350)
        self.statusRow = pn.Row(pn.pane.Markdown("", margin=0))

    def _make_output_params(self, start_path):
        return OutputParams(start_path=start_path)

    def _make_time_picker(self):
        return TimePicker()

    # ------------------------------------------------------------------ #
    #  Compute settings                                                    #
    # ------------------------------------------------------------------ #

    def _update_gpu_options(self, event=None):
        system = SYSTEMS[self.systemPicker.value]
        types = list(system.gpu_types)
        if self.gpuTypePicker.value not in types:
            self.gpuTypePicker.options = types
            self.gpuTypePicker.value = types[0]
        else:
            self.gpuTypePicker.options = types
        # More GPUs than models would sit idle: each GPU runs whole models.
        most = max(1, min(system.max_gpus, len(self.modelPicker.value)))
        count = min(self.gpuCountPicker.value or 1, most)
        self.gpuCountPicker.options = list(range(1, most + 1))
        self.gpuCountPicker.value = count
        self._update_walltime()

    def _estimates(self):
        start, end = self.timePicker.start_time, self.timePicker.end_date
        if start is None or end is None:
            return []
        return [(m, model_seconds(m, start, end)) for m in self.modelPicker.value]

    def _walltime(self):
        """(seconds to request, whether the system's limit cut it short)."""
        return estimate_walltime(self._estimates(), self.gpuCountPicker.value or 1,
                                 self.systemPicker.value)

    def _update_walltime(self, event=None):
        if not self.modelPicker.value:
            self.walltimeLabel.value = "N/A"
            return
        seconds, capped = self._walltime()
        text = format_duration(seconds)
        if capped:
            text += (f" ({self.systemPicker.value}'s limit; this run may need "
                     "longer - shorten it or use more GPUs)")
        self.walltimeLabel.value = text

    # ------------------------------------------------------------------ #
    #  Runner helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_runners(self):
        """Return list of (model_name, runner) for every selected model."""
        runners = []
        for model in self.modelPicker.value:
            if model == 'WXFormerGen2':
                runners.append((model, MilesCreditGen2Runner()))
            elif model in MILES_CREDIT_MODELS:
                runners.append((model, MilesCreditRunner()))
            elif model in EARTH2STUDIO_MODELS:
                runners.append((model, Earth2StudioRunner()))
        return runners

    def _build_config(self, model: str) -> dict:
        return {
            "simulation_name": self.outputParams.simulationNamePicker.value_input,
            "start_time":      self.timePicker.start_time,
            "end_time":        self.timePicker.end_date,
            "timestep":        self.timePicker.timestep,
            "ua_vars":         self.outputParams.UAVars.value,
            "surface_vars":    self.outputParams.surfaceVars.value,
            "output_path":     str(
                Path(self.outputParams.pathDisplay.value)
                / self.outputParams.simulationNamePicker.value_input
            ),
            "output_dir":      self.outputParams.current_path_val,
            "model":           model,
            "file_stem":       model_file_stem(
                self.outputParams.simulationNamePicker.value_input, model
            ),
        }

    # ------------------------------------------------------------------ #
    #  UI event handlers                                                   #
    # ------------------------------------------------------------------ #

    def _on_run_click(self, event):
        runners = self._get_runners()

        if not runners:
            self._set_single_log("Error: No recognized model selected.")
            return

        self._clear_highlight(self.outputParams.simulationNamePicker)

        # Pre-validate all runners before touching the UI
        for model, runner in runners:
            config = self._build_config(model)
            error = runner.validate(config)
            if error:
                self._set_single_log(f"[{model}] {error}")
                if not config["simulation_name"]:
                    self._highlight_error(self.outputParams.simulationNamePicker)
                return

        if not self._account:
            self._set_single_log(
                "Error: No project account to charge the inference job to. "
                "InferStudio uses the one chosen when it was started from NCAR "
                "Open OnDemand; outside Open OnDemand, set INFERSTUDIO_ACCOUNT "
                "before starting it.")
            return

        # Build tabs, spinners, and status indicators — one per model
        self._log_widgets = {}
        self._spinners = {}
        self._status_widgets = {}
        self._failed_models = []
        self.outputTabs.objects = []
        self.statusRow.objects = []

        for model, _ in runners:
            # No name: the log's tab already shows the model.
            widget = pn.widgets.TextAreaInput(
                value="",
                sizing_mode="stretch_width",
                height=300,
            )
            widget.jscallback(value=LOG_AUTOSCROLL_JS)
            spinner = pn.indicators.LoadingSpinner(
                width=25, height=25, value=False, color="primary", visible=False
            )
            pane = pn.pane.HTML(self._status_html(model, "pending"), width=100)
            self._log_widgets[model] = widget
            self._spinners[model] = spinner
            self._status_widgets[model] = pane
            self.statusRow.append(pane)
            self.outputTabs.append((model, pn.Column(spinner, widget, sizing_mode="stretch_width", height=330)))

        self.spinner.value = True
        self.spinner.visible = True
        self.inferenceButton.disabled = True
        self.cancelButton.disabled = False
        self.jobLabel.value = "Submitting..."
        self.queueLabel.value = ""
        self.elapsedLabel.value = ""
        self.completionLabel.value = ""
        self.completionPathLabel.value = ""

        run = self._run = _Run(
            system=self.systemPicker.value,
            gpu_type=SYSTEMS[self.systemPicker.value].gpu_types[self.gpuTypePicker.value],
            n_gpus=self.gpuCountPicker.value or 1,
        )
        self._doc = pn.state.curdoc

        # Time in queue runs from submission to the job starting, Elapsed
        # from then to the end, so a long wait for GPUs isn't mistaken for
        # slow inference.
        def _tick_update():
            if run.submitted_at is None:
                return
            now = time.time()
            queue_end = run.started_at or run.ended_at or now
            self.queueLabel.value = format_duration(queue_end - run.submitted_at)
            if run.started_at is not None:
                self.elapsedLabel.value = format_duration((run.ended_at or now) - run.started_at)
            else:
                self.elapsedLabel.value = "waiting for the job to start"

        if self._periodic_cb is not None:
            self._periodic_cb.stop()
        self._periodic_cb = pn.state.add_periodic_callback(_tick_update, period=1000)
        run.tick = _tick_update

        def _all_done():
            sim_dir = Path(self.outputParams.pathDisplay.value) / self.outputParams.simulationNamePicker.value_input

            def _finish():
                if run.cancel.is_set():
                    # _on_cancel_click already reset the button/spinner
                    # the moment Cancel was clicked -- this run never
                    # completed, so don't overwrite that with a
                    # misleading "Files written to ..." success message.
                    return
                try:
                    if self._periodic_cb is not None:
                        self._periodic_cb.stop()
                    _tick_update()
                    if self._failed_models:
                        self.completionLabel.value = "Completed with errors"
                        self.completionPathLabel.value = (
                            f"Failed: {', '.join(self._failed_models)} "
                            "— see that model's log tab for details."
                        )
                    else:
                        self.completionLabel.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.completionPathLabel.value = f"Files written to {sim_dir}"
                except Exception:
                    traceback.print_exc()
                finally:
                    # Always reset the Run/Cancel controls, no matter what
                    # happens below -- setting outputDirectory synchronously
                    # runs the Visualization tab's dataset-loading watcher
                    # (scan_simulation_suite, dataset-browser wiring, the tab
                    # switch), and a failure there must not leave the
                    # inference tab itself looking like it's still running.
                    self.spinner.value = False
                    self.spinner.visible = False
                    self.inferenceButton.disabled = False
                    self.cancelButton.disabled = True

                if self._failed_models:
                    # At least one model failed outright -- stay on the
                    # Inference tab (skip the outputDirectory trigger below,
                    # so app_layout's Visualization-tab switch never runs)
                    # and make sure the failure is impossible to miss rather
                    # than only visible as a red status icon on that model's
                    # tab. The suite can still be picked up later via "Load
                    # Existing Suite" if a partial result is worth keeping.
                    if pn.state.notifications:
                        plural = "s" if len(self._failed_models) > 1 else ""
                        pn.state.notifications.error(
                            f"Inference failed for model{plural}: "
                            f"{', '.join(self._failed_models)}. See that "
                            "model's log tab for details.",
                            duration=0,
                        )
                    return

                try:
                    # A plain assignment only fires the outputDirectory
                    # watcher (app_layout._on_new_output) when the value
                    # actually changes -- which covers the normal case of a
                    # fresh simulation_name. Re-running the exact same
                    # simulation_name back-to-back would otherwise silently
                    # not reload it, so that one case falls back to an
                    # explicit trigger() instead. Doing BOTH unconditionally
                    # (the previous code) fired the watcher twice per run --
                    # duplicating the scan and posting the completion
                    # notification twice.
                    new_dir = str(sim_dir)
                    if new_dir == self.outputDirectory:
                        self.param.trigger('outputDirectory')
                    else:
                        self.outputDirectory = new_dir
                except Exception:
                    traceback.print_exc()
                    if pn.state.notifications:
                        pn.state.notifications.error(
                            f"Inference finished, but failed to load {sim_dir} "
                            "into the Visualization tab. See server log for details.",
                            duration=0,
                        )

            self._ui(_finish)

        def _run_job():
            try:
                self._run_job(run, runners)
                _all_done()
            except Exception:
                tb = traceback.format_exc()
                if run.job is not None:
                    run.job.delete()

                def _report():
                    for model in self._log_widgets:
                        self._append_log(model, f"\n\n[INTERNAL ERROR]\n{tb}\n")
                    if self._periodic_cb is not None:
                        self._periodic_cb.stop()
                    self.spinner.value = False
                    self.spinner.visible = False
                    self.inferenceButton.disabled = False
                    self.cancelButton.disabled = True

                self._ui(_report)

        threading.Thread(target=_run_job, daemon=True).start()

    def _on_cancel_click(self, event):
        run = self._run
        if run is None:
            return
        run.cancel.set()
        if run.ended_at is None:
            run.ended_at = time.time()
        job = run.job
        if job is not None:
            # qdel can take a few seconds; don't hold up the page for it.
            def _delete():
                error = job.delete()
                for model in self._log_widgets:
                    self._append_log(model, f"\n\nCancelled by user: deleted PBS job {job.job_id}"
                                     + (f" ({error})" if error else "") + ".\n")
            threading.Thread(target=_delete, daemon=True).start()

        # Reset the shared Run/Cancel controls right away, rather than
        # waiting for the job to be gone from PBS.
        if self._periodic_cb is not None:
            self._periodic_cb.stop()
        if run.tick is not None:
            run.tick()
        self.spinner.value = False
        self.spinner.visible = False
        self.inferenceButton.disabled = False
        self.cancelButton.disabled = True
        self.completionLabel.value = "Cancelled"
        self.completionPathLabel.value = ""

    # ------------------------------------------------------------------ #
    #  The PBS job                                                         #
    # ------------------------------------------------------------------ #

    def _ui(self, fn):
        """Schedule a UI-mutating callable safely on the document's own thread."""
        if self._doc is not None:
            self._doc.add_next_tick_callback(fn)
        else:
            fn()

    def _append_log(self, model: str, text: str):
        widget = self._log_widgets.get(model)
        if widget is None:
            return
        def _do_append():
            widget.value += text
        self._ui(_do_append)

    def _set_status(self, model: str, state: str):
        def _update():
            spinner = self._spinners.get(model)
            if spinner is not None:
                spinner.value = spinner.visible = state == "running"
            pane = self._status_widgets.get(model)
            if pane is not None:
                pane.object = self._status_html(model, state)
            if state == "running" and model in self._log_widgets:
                self.outputTabs.active = list(self._log_widgets).index(model)
        self._ui(_update)

    def _set_job_label(self, text: str):
        def _update():
            self.jobLabel.value = text
        self._ui(_update)

    def _run_job(self, run, runners):
        """Write each model's command, submit them as one PBS job and follow
        it until it ends. Runs in a background thread."""
        start, end = self.timePicker.start_time, self.timePicker.end_date
        runs = []
        for model, runner in runners:
            config = self._build_config(model)
            model_output = Path(config["output_path"]) / model
            try:
                model_output.mkdir(parents=True, exist_ok=True)
                config["output_path"] = str(model_output)
                config.update(runner.prepare(config))
                cmd = runner.build_cmd(config)
            except Exception as e:
                self._append_log(model, f"Error during setup: {e}\n")
                self._failed_models.append(model)
                self._set_status(model, "error")
                continue
            # The job writes the log next to the results, so warnings (e.g.
            # missing input variables) are still visible after the session.
            runs.append(ModelRun(model, cmd, model_output / f"{model}_run.log",
                                 model_seconds(model, start, end)))
        if not runs or run.cancel.is_set():
            self._set_job_label("not submitted")
            return

        job = PBSJob.create(self.outputParams.simulationNamePicker.value_input, run.system)
        # Fewer models than GPUs (can't happen from the picker, but a model
        # can fail setup above) means fewer GPUs.
        n_gpus = min(run.n_gpus, len(runs))
        walltime, _ = estimate_walltime([(r.model, r.seconds) for r in runs], n_gpus, run.system)
        script = job.write_script(runs, self._account, run.gpu_type, n_gpus,
                                  walltime, os.getcwd())
        gpus = f"{n_gpus} x {run.gpu_type}"
        config = self._build_config(runs[0].model)
        usage = dict(
            tab=type(self).__name__.removesuffix("Tab").lower(),
            models=[r.model for r in runs],
            setup_failed=list(self._failed_models),
            system=run.system, gpu_type=run.gpu_type, gpus=n_gpus,
            walltime_s=walltime, start_time=start, end_time=end,
            timestep=config["timestep"],
            ua_vars=config["ua_vars"], surface_vars=config["surface_vars"],
            **({"forcing_mode": config["forcing_mode"]} if "forcing_mode" in config else {}),
        )
        run.submitted_at = time.time()
        try:
            job.submit(script)
        except RuntimeError as e:
            log_event("inference_submit", outcome="error", error=short_error(e), **usage)
            run.submitted_at = None
            for r in runs:
                self._append_log(r.model, f"Could not submit the PBS job: {e}\nJob script: {script}\n")
                self._failed_models.append(r.model)
                self._set_status(r.model, "error")
            self._set_job_label("submission failed")
            return
        run.job = job
        log_event("inference_submit", outcome="submitted", job=job.job_id, **usage)
        if run.cancel.is_set():
            job.delete()
            log_event("inference_end", job=job.job_id, outcome="cancelled",
                      models={r.model: "cancelled" for r in runs})
            return

        self._set_job_label(f"{job.job_id} - queued on {run.system} ({gpus})")
        for r in runs:
            self._append_log(
                r.model,
                f"Submitted PBS job {job.job_id} to {run.system}: {gpus}, "
                f"{format_duration(walltime)} wall time, account {self._account}.\n"
                f"Job files: {job.job_dir}\n"
                "Waiting in the queue...\n")
        states = self._follow(run, job, runs)

        # Models still pending or running when the job ended were cancelled
        # or cut short.
        cut = "cancelled" if run.cancel.is_set() else "error"
        outcomes = {m: cut if st in ("pending", "running") else st
                    for m, st in states.items()}
        log_event(
            "inference_end", job=job.job_id,
            outcome=("cancelled" if run.cancel.is_set() else
                     "done" if all(o == "done" for o in outcomes.values()) else "error"),
            models=outcomes,
            queue_wait_s=round((run.started_at or run.ended_at) - run.submitted_at),
            run_s=round(run.ended_at - run.started_at) if run.started_at else None,
            walltime_s=walltime,
        )

    # How often to ask PBS about the job; the files it writes are checked
    # every second.
    QSTAT_INTERVAL = 15
    # Once every model has finished, how long the job gets to end by
    # itself before it's deleted.
    EXIT_GRACE = 30

    def _follow(self, run, job, runs):
        tails = {r.model: LogTail(r.log_path) for r in runs}
        states = {r.model: "pending" for r in runs}
        last_qstat = 0.0
        all_done_at = None
        last = None

        def check(m):
            """Pass on the model's new output, and whether it has started
            or finished."""
            if states[m] == "pending" and job.model_started_at(m) is not None:
                states[m] = "running"
                self._set_status(m, "running")
            if states[m] != "running":
                return
            code = job.model_exit(m)
            text = tails[m].read()
            if text:
                self._append_log(m, text)
            if code is not None:
                states[m] = "done" if code == 0 else "error"
                if code != 0:
                    self._failed_models.append(m)
                self._append_log(m, "\nDone.\n" if code == 0 else f"\nExited with code {code}\n")
                self._set_status(m, states[m])

        while not run.cancel.is_set():
            now = time.time()
            if run.started_at is None:
                started = job.started_at()
                if started is not None:
                    run.started_at = min(max(started, run.submitted_at), now)
                    log_event("inference_start", job=job.job_id,
                              queue_wait_s=round(run.started_at - run.submitted_at),
                              exec_host=last.exec_host if last and last.exec_host else None)
                    host = last.exec_host if last and last.exec_host else run.system
                    self._set_job_label(f"{job.job_id} - running on {host}")

            for r in runs:
                check(r.model)

            finished = all(s in ("done", "error") for s in states.values())
            if finished and all_done_at is None:
                all_done_at = now
            interval = 5 if finished else self.QSTAT_INTERVAL
            if now - last_qstat >= interval:
                last_qstat = now
                status = job.status()
                if status is None or status.state == 'F':
                    last = status or last
                    break
                if status.state != '?':
                    last = status
                if run.started_at is None and status.state in ('Q', 'H', 'W'):
                    why = f" - {status.comment}" if status.comment else ""
                    self._set_job_label(f"{job.job_id} - {_STATE_NAMES.get(status.state, status.state)}"
                                        f" on {run.system}{why}")
                if finished and now - all_done_at > self.EXIT_GRACE:
                    # Every model is done; don't let a straggling process
                    # keep the GPUs.
                    job.delete()
            run.cancel.wait(1)

        run.ended_at = run.ended_at or time.time()
        if run.started_at is None and not run.cancel.is_set():
            # Ended without ever starting (deleted from outside, held, ...)
            run.started_at = run.ended_at

        # Whatever the models wrote, and any that finished, after the last
        # look.
        for r in runs:
            check(r.model)

        unfinished = [m for m, s in states.items() if s in ("pending", "running")]
        if not unfinished:
            self._set_job_label(f"{job.job_id} - finished")
            return states
        if run.cancel.is_set():
            for m in unfinished:
                self._set_status(m, "cancelled")
            self._set_job_label(f"{job.job_id} - cancelled")
            return states

        why = "The PBS job ended before this model finished"
        if last is not None and last.comment:
            why += f" ({last.comment})"
        why += "."
        if last is not None and last.exit_status is not None:
            why += f" Job exit status: {last.exit_status}."
        output = job.pbs_output().strip()
        if output:
            why += "\n\nPBS job output:\n" + "\n".join(output.splitlines()[-40:])
        for m in unfinished:
            self._failed_models.append(m)
            self._append_log(m, f"\n{why}\n")
            self._set_status(m, "error")
        self._set_job_label(f"{job.job_id} - ended")
        return states


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

    def _status_html(self, model: str, state: str) -> str:
        symbol = {"pending": "◷", "running": "⟳", "done": "✓", "error": "✗", "cancelled": "⊘"}[state]
        color  = {"pending": "#cccccc", "running": "#888888", "done": "#2ecc71", "error": "#e74c3c", "cancelled": "#999999"}[state]
        return (
            f'<div style="text-align:center;line-height:1.4">'
            f'<span style="font-size:11px;color:#555">{model}</span><br>'
            f'<span style="font-size:20px;color:{color}">{symbol}</span>'
            f'</div>'
        )

    def _update_default_sim_name(self, event=None):
        self.outputParams.set_default_simulation_name(self.modelPicker.value)

    def _highlight_error(self, widget):
        styles = dict(widget.styles or {})
        styles.update({'border': '2px solid #e74c3c', 'border-radius': '4px'})
        widget.styles = styles

    def _clear_highlight(self, widget):
        styles = dict(widget.styles or {})
        styles.pop('border', None)
        styles.pop('border-radius', None)
        widget.styles = styles

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    def panel(self):
        compute = pn.WidgetBox(
            "### Compute Settings",
            pn.pane.Markdown(
                "The models run in their own PBS batch job, which ends as "
                "soon as they finish.",
                margin=(0, 10, 0, 10),
            ),
            pn.Row(*(
                pn.Column(pn.pane.Markdown(label, margin=(0, 10, -5, 10)), widget)
                for label, widget in [("System", self.systemPicker),
                                      ("GPU type", self.gpuTypePicker),
                                      ("GPUs", self.gpuCountPicker)]
            )),
            pn.Row(self.walltimeLabel, self.accountLabel),
            sizing_mode='stretch_width',
        )
        model = pn.WidgetBox(
            '### AI Model',
            self.modelPicker.panel(),
        )
        # A 2x2 grid of settings, so the Launcher's logs start higher up.
        # Each box fills its cell, so the two in a row match in height, and
        # gets an outline and an accent edge to set it apart.
        boxes = [model, self.outputParams.panel(), self.timePicker.panel(), compute]
        for box in boxes:
            box.sizing_mode = 'stretch_both'
            box.styles = dict(box.styles or {}, **SETTINGS_BOX_STYLES)
        settings = pn.GridBox(*boxes, ncols=2, sizing_mode='stretch_width')
        return pn.Column(
            settings,
            pn.WidgetBox(
                "### Launcher",
                pn.Row(
                    self.inferenceButton, self.cancelButton,
                    self.spinner, self.statusRow,
                ),
                pn.FlexBox(self.jobLabel, self.queueLabel, self.elapsedLabel,
                           self.completionLabel, flex_wrap='wrap'),
                self.completionPathLabel,
                self.outputTabs,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_both',
            min_height=300,
        )


class WeatherTab(ModelTab):
    MODELS = WEATHER_MODEL_LIST
    DEFAULT_MODELS = ['AIFS', 'Aurora']


# CAMulator diagnostics offered on the Climate tab, button label -> output
# name. Names ending in digits (tp06, e06) would be read as pressure levels
# by the Visualization tab, so they aren't offered yet.
CLIMATE_SURFACE_VARS = {
    'SP': 'SP',
    't2m': 't2m',
    'Skin T': 'skt',
    '10m Wind': 'ws10m',
    'OLR': 'mtuwlwrf',
    'High Cloud': 'hcc',
    'Mid Cloud': 'mcc',
    'Low Cloud': 'lcc',
}


class ClimateTab(ModelTab):
    MODELS = CLIMATE_MODEL_LIST
    DEFAULT_MODELS = ['CAMulator']

    def _make_output_params(self, start_path):
        return OutputParams(
            start_path=start_path,
            surface_options=CLIMATE_SURFACE_VARS,
            surface_default=['SP', 't2m'],
        )

    def _make_time_picker(self):
        return ClimateTimePicker()

    def _build_config(self, model: str) -> dict:
        config = super()._build_config(model)
        config["forcing_mode"] = self.timePicker.forcing_mode
        return config


class InferenceTab(param.Parameterized):
    """The Inference tab: a Weather sub-tab for the weather models, and a
    Climate sub-tab for climate models such as CAMulator.

    outputDirectory is set whenever either sub-tab finishes a run, so the
    app can load the new suite into the Visualization tab."""

    outputDirectory = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self.weatherTab = WeatherTab()
        self.climateTab = ClimateTab()
        for tab in (self.weatherTab, self.climateTab):
            tab.param.watch(self._forward_output, 'outputDirectory')

    def _forward_output(self, event):
        # A sub-tab triggers outputDirectory even when it's unchanged (a
        # re-run of the same simulation name), so this does the same.
        if event.new == self.outputDirectory:
            self.param.trigger('outputDirectory')
        else:
            self.outputDirectory = event.new

    def panel(self):
        # Styled like the app's main tabs; see tabStyles for the title's
        # second line.
        return pn.Tabs(
            ("Weather", self.weatherTab.panel()),
            ("Climate\n(experimental!)", self.climateTab.panel()),
            stylesheets=[TAB_HEADER_CSS],
            sizing_mode='stretch_both',
            min_height=300,
        )
