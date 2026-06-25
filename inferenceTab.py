import panel as pn
import param
import subprocess
import threading
import os
import shutil
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
            margin=(0,5,5,0)
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

        self.outputLog = pn.widgets.TextAreaInput(
            name="Output Log",
            value="",
            #height=200,
            sizing_mode="stretch_both",
        )

        self.elapsedLabel = pn.widgets.StaticText(name="Elapsed", value="N/A")
        self.completionLabel = pn.widgets.StaticText(name="Completed at", value="N/A")
        self.commandRunner = CommandRunner()
        self._process = None

    def _get_runners(self):
        runners = []
        for model in self.modelPicker.value:
            if model in MILES_CREDIT_MODELS:
                runners.append((model, MilesCreditRunner()))
            elif model in EARTH2STUDIO_MODELS:
                runners.append((model, Earth2StudioRunner()))
        return runners

    #def _get_runner(self):
    #    selected = set(self.modelPicker.value)
    #    if selected & MILES_CREDIT_MODELS:
    #        return MilesCreditRunner()
    #    elif selected & EARTH2STUDIO_MODELS:
    #        return Earth2StudioRunner()
    #    return None

    #def _replaceParams(self):
    #    with open('model_predict_casper.yml', 'r') as f:
    #        content = f.read()

    #    content = content.replace(
    #        'forecast_start_time: "2025-07-02 00:00:00"',
    #        f'forecast_start_time: "{self.timePicker.startDatePicker.value}"'
    #    )
    #    content = content.replace(
    #        'forecast_end_time: "2025-07-02 02:00:00"',
    #        f'forecast_end_time: "{self.timePicker.endDatePicker.value}"'
    #    )
    #    content = content.replace(
    #        'forecast_timestep: "1h"',
    #        f'forecast_timestep: "{self.timePicker.incrementButtons.value}"'
    #    )
    #    content = content.replace(
    #        "variables: ['U','V','T','Q']",
    #        f"variables: {self.outputParams.UAVars.value}"
    #    )
    #    content = content.replace(
    #        "surface_variables: ['SP','t2m','V500','U500','T500','Z500','Q500']",
    #        f"surface_variables: {self.outputParams.surfaceVars.value}"
    #    )
    #    content = content.replace(
    #        "save_forecast: '/glade/derecho/scratch/pearse/CREDIT/RAW_OUTPUT/wxformer_1h_gfs_demo/'",
    #        f"save_forecast: '{self.outputParams.pathDisplay.value}'"
    #    )
    #    
    #    self.configFile = self.outputParams.current_path_val + '/' + self.outputParams.simulationNamePicker.value + '.yml'
    #    with open(self.configFile, 'w') as f:
    #        f.write(content)

    def _build_config(self, model: str):
        return {
            "simulation_name": self.outputParams.simulationNamePicker.value,
            "start_time":      self.timePicker.startDatePicker.value,
            "end_time":        self.timePicker.endDatePicker.value,
            "timestep":        self.timePicker.incrementButtons.value,
            "ua_vars":         self.outputParams.UAVars.value,
            "surface_vars":    self.outputParams.surfaceVars.value,
            "output_path":     self.outputParams.pathDisplay.value,
            "output_dir":      self.outputParams.current_path_val,
            "model":           model,   # single string now
        }

    #def _build_config(self):
    #    return {
    #        "simulation_name": self.outputParams.simulationNamePicker.value,
    #        "start_time":      self.timePicker.startDatePicker.value,
    #        "end_time":        self.timePicker.endDatePicker.value,
    #        "timestep":        self.timePicker.incrementButtons.value,
    #        "ua_vars":         self.outputParams.UAVars.value,
    #        "surface_vars":    self.outputParams.surfaceVars.value,
    #        "output_path":     self.outputParams.pathDisplay.value,
    #        "output_dir":      self.outputParams.current_path_val,
    #        "model":           self.modelPicker.value,
    #    }

    def _on_run_click(self, event):
        print("_on_run_click fired", flush=True)
        runners = self._get_runners()
        if not runners:
            self.outputLog.value = "Error: No recognized model selected."
            return
    
        # Pre-validate all runners before starting anything
        for model, runner in runners:
            config = self._build_config(model)
            error = runner.validate(config)
            if error:
                self.outputLog.value = f"[{model}] {error}"
                return
    
        self.spinner.value = True
        self.spinner.visible = True
        self.inferenceButton.disabled = True
        self.cancelButton.disabled = False
    
        def _prepare_and_run():
            self._execute_all(runners)
    
        threading.Thread(target=_prepare_and_run).start()

    #def _on_run_click(self, event):
    #    print("_on_run_click fired", flush=True)
    #    config = self._build_config()
    #    print(f"config: {config}", flush=True)
    #    runner = self._get_runner()
    #    print(f"runner: {runner}", flush=True)
    #    if runner is None:
    #        self.outputLog.value = "Error: No recognized model selected."
    #        return
 
    #    error = runner.validate(config)
    #    print(f"error: {error}", flush=True)
    #    if error:
    #        self.outputLog.value = error
    #        return

    #    self.spinner.value = True
    #    self.spinner.visible = True
    #    self.inferenceButton.disabled = True
    #    self.cancelButton.disabled = False

    #    def _prepare_and_run():
    #        try:
    #            prepared = runner.prepare(config)
    #            config.update(prepared)
    #            cmd = runner.build_cmd(config)
    #        except Exception as e:
    #            self.outputLog.value = f"Error during setup: {str(e)}"
    #            self.spinner.value = False
    #            self.spinner.visible = False
    #            self.inferenceButton.disabled = False
    #            self.cancelButton.disabled = True
    #            return
    #        self._execute(cmd)
    #    #def _prepare_and_run(): 
    #    #    prepared = runner.prepare(config)
    #    #    config.update(prepared)
    #    #    cmd = runner.build_cmd(config)
    #    #    self._execute(cmd)
 
    #    thread = threading.Thread(target=_prepare_and_run)
    #    thread.start()
 
    #def _on_run_click(self, event):
    #    if not self.outputParams.simulationNamePicker.value.strip():
    #        self.outputLog.value = "Error: Please enter a simulation name."
    #        return

    #    self._replaceParams()

    #    #cmd = f"""python /glade/work/pearse/credit-panel/miles-credit/applications/gfs_init.py -c {self.configFile} &&
    #    #          python /glade/work/pearse/credit-panel/miles-credit/applications/rollout_realtime.py -c {self.configFile}"""
    #    #cmd = f"""python /glade/work/pearse/credit-panel/miles-credit/applications/rollout_realtime.py -c {self.configFile}"""
    #    cmd = f"""python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/gfs_init.py -c {self.configFile} &&
    #              python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/rollout_realtime.py -c {self.configFile}"""
    #    #cmd = f"""python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/rollout_realtime.py -c {self.configFile}"""

    #    if not cmd: return

    #    # UI updates happen immediately here
    #    self.spinner.value = True
    #    self.spinner.visible = True
    #    self.inferenceButton.disabled = True
    #    self.cancelButton.disabled = False

    #    # Launch the actual subprocess in the background
    #    thread = threading.Thread(target=self._execute, args=(cmd,))
    #    thread.start()

    def _on_cancel_click(self, event):
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except Exception:
                self._process.terminate()
            self.outputLog.value += "\n\nCancelled by user."
        self.cancelButton.disabled = True

    def _execute_all(self, runners):
        self.outputLog.value = ""
        self.elapsedLabel.value = ""
        self.completionLabel.value = ""
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
    
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
    
        cancelled = False
        try:
            for i, (model, runner) in enumerate(runners, 1):
                if cancelled:
                    break
    
                self.outputLog.value += f"\n{'='*60}\n[{i}/{len(runners)}] {model}\n{'='*60}\n"
    
                config = self._build_config(model)
    
                # Each model writes into <output_path>/<model_name>/
                base_output = Path(config["output_path"])
                model_output = base_output / model
                model_output.mkdir(parents=True, exist_ok=True)
                config["output_path"] = str(model_output)
    
                try:
                    prepared = runner.prepare(config)
                    config.update(prepared)
                    cmd = runner.build_cmd(config)
                except Exception as e:
                    self.outputLog.value += f"Error during setup: {e}\n"
                    continue   # try remaining models
    
                try:
                    self._process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        start_new_session=True,
                    )
                    for line in self._process.stdout:
                        self.outputLog.value += line
                    self._process.wait()
    
                    if self._process.returncode != 0:
                        self.outputLog.value += f"\n{model} exited with code {self._process.returncode}\n"
    
                except Exception as e:
                    self.outputLog.value += f"Error running {model}: {e}\n"
    
                finally:
                    # If cancel was clicked mid-model the process is already dead;
                    # check so we don't try subsequent models.
                    if self._process is not None and self._process.returncode is None:
                        cancelled = True
                    self._process = None
    
            if cancelled:
                self.outputLog.value += "\nCancelled — remaining models skipped.\n"
    
            elapsed = time.time() - overall_start
            mins, secs = divmod(int(elapsed), 60)
            self.elapsedLabel.value = f"{mins}m {secs}s"
            self.completionLabel.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        finally:
            self._timer_running = False
            timer_thread.join()
            self._process = None
            self.spinner.value = False
            self.spinner.visible = False
            self.inferenceButton.disabled = False
            self.cancelButton.disabled = True

    #def _execute(self, cmd):
    #    env = os.environ.copy()
    #    env["PYTHONUNBUFFERED"] = "1"

    #    self.outputLog.value = ""
    #    self.elapsedLabel.value = ""
    #    self.completionLabel.value = ""
    #    start_time = time.time()
    #    self._timer_running = True

    #    def _tick():
    #        while self._timer_running:
    #            elapsed = time.time() - start_time
    #            mins, secs = divmod(int(elapsed), 60)
    #            self.elapsedLabel.value = f"{mins}m {secs}s"
    #            time.sleep(1)

    #    timer_thread = threading.Thread(target=_tick, daemon=True)
    #    timer_thread.start()

    #    try:
    #        self._process = subprocess.Popen(
    #            cmd,
    #            shell=True,
    #            stdout=subprocess.PIPE,
    #            stderr=subprocess.STDOUT,
    #            text=True,
    #            env=env,
    #            start_new_session=True
    #        )

    #        for line in self._process.stdout:
    #            self.outputLog.value += line

    #        self._process.wait()

    #        elapsed = time.time() - start_time
    #        mins, secs = divmod(int(elapsed), 60)
    #        self.elapsedLabel.value = f"{mins}m {secs}s"
    #        self.completionLabel.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #        if self._process.returncode != 0:
    #            self.outputLog.value += f"\nProcess exited with code {self._process.returncode}"

    #    except Exception as e:
    #        self.outputLog.value = f"Error: {str(e)}"
    #    finally:
    #        self._timer_running = False
    #        timer_thread.join()
    #        self._process = None
    #        self.spinner.value = False
    #        self.spinner.visible = False
    #        self.inferenceButton.disabled = False
    #        self.cancelButton.disabled = True
    
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
                self.outputLog,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_both',
            min_height=300
        )
