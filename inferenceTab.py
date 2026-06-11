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

from directoryPicker import DirectoryPicker
from timePicker import TimePicker
from commandRunner import CommandRunner

class InferenceTab(param.Parameterized):
    startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))

    def __init__(self, **params):
        super().__init__(**params)

        #self.modelPicker = pn.widgets.RadioButtonGroup(
        self.modelPicker = pn.widgets.CheckButtonGroup(
            name="Select Model",
            value=['WXFormer', 'AIFS'],
            options=['WXFormer', 'AIFS', 'Aurora', 'FourCastNet3', 'GraphCast', 'Pangu', 'SFNO'],
            button_type='primary',
            button_style='outline',
            margin=(0,5,5,0)
        )

        self.outputDirPicker = DirectoryPicker(start_path=Path.home())

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

    def _replaceParams(self):
        with open('model_predict_casper.yml', 'r') as f:
            content = f.read()

        content = content.replace(
            'forecast_start_time: "2025-07-02 00:00:00"',
            f'forecast_start_time: "{self.timePicker.startDatePicker.value}"'
        )
        content = content.replace(
            'forecast_end_time: "2025-07-02 02:00:00"',
            f'forecast_end_time: "{self.timePicker.endDatePicker.value}"'
        )
        content = content.replace(
            'forecast_timestep: "1h"',
            f'forecast_timestep: "{self.timePicker.incrementButtons.value}"'
        )
        content = content.replace(
            "variables: ['U','V','T','Q']",
            f"variables: {self.outputDirPicker.UAVars.value}"
        )
        content = content.replace(
            "surface_variables: ['SP','t2m','V500','U500','T500','Z500','Q500']",
            f"surface_variables: {self.outputDirPicker.surfaceVars.value}"
        )
        content = content.replace(
            "save_forecast: '/glade/derecho/scratch/pearse/CREDIT/RAW_OUTPUT/wxformer_1h_gfs_demo/'",
            f"save_forecast: '{self.outputDirPicker.pathDisplay.value}'"
        )
        
        self.configFile = self.outputDirPicker.current_path_val + '/' + self.outputDirPicker.simulationNamePicker.value + '.yml'
        with open(self.configFile, 'w') as f:
            f.write(content)
 
    def _on_run_click(self, event):
        if not self.outputDirPicker.simulationNamePicker.value.strip():
            self.outputLog.value = "Error: Please enter a simulation name."
            return

        self._replaceParams()

        #cmd = f"""python /glade/work/pearse/credit-panel/miles-credit/applications/gfs_init.py -c {self.configFile} &&
        #          python /glade/work/pearse/credit-panel/miles-credit/applications/rollout_realtime.py -c {self.configFile}"""
        #cmd = f"""python /glade/work/pearse/credit-panel/miles-credit/applications/rollout_realtime.py -c {self.configFile}"""
        cmd = f"""python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/gfs_init.py -c {self.configFile} &&
                  python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/rollout_realtime.py -c {self.configFile}"""
        #cmd = f"""python $CONDA_PREFIX/lib/python3.12/site-packages/credit/applications/rollout_realtime.py -c {self.configFile}"""

        if not cmd: return

        # UI updates happen immediately here
        self.spinner.value = True
        self.spinner.visible = True
        self.inferenceButton.disabled = True
        self.cancelButton.disabled = False

        # Launch the actual subprocess in the background
        thread = threading.Thread(target=self._execute, args=(cmd,))
        thread.start()

    def _on_cancel_click(self, event):
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except Exception:
                self._process.terminate()
            self.outputLog.value += "\n\nCancelled by user."
        self.cancelButton.disabled = True

    def _execute(self, cmd):
        #env = os.environ.copy()
        #env["PYTHONUNBUFFERED"] = "1"

        #self.outputLog.value = subprocess.run(
        #    "echo CONDA_PREFIX=$CONDA_PREFIX", shell=True, env=env,
        #    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        #).stdout
        #return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.outputLog.value = ""
        self.elapsedLabel.value = ""
        self.completionLabel.value = ""
        start_time = time.time()
        self._timer_running = True

        def _tick():
            while self._timer_running:
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                self.elapsedLabel.value = f"{mins}m {secs}s"
                time.sleep(1)

        timer_thread = threading.Thread(target=_tick, daemon=True)
        timer_thread.start()

        try:
            self._process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True
            )

            for line in self._process.stdout:
                self.outputLog.value += line

            self._process.wait()

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            self.elapsedLabel.value = f"{mins}m {secs}s"
            self.completionLabel.value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if self._process.returncode != 0:
                self.outputLog.value += f"\nProcess exited with code {self._process.returncode}"
        except Exception as e:
            self.outputLog.value = f"Error: {str(e)}"
        finally:
            self._timer_running = False
            timer_thread.join()
            self._process = None
            self.spinner.value = False
            self.spinner.visible = False
            self.inferenceButton.disabled = False
            self.cancelButton.disabled = True
    
    def panel(self):
        return pn.Column(
            pn.WidgetBox(
                '# AI Model',
                self.modelPicker,
                sizing_mode='stretch_width',
            ),
            self.outputDirPicker.panel(),
            self.timePicker.panel,
            pn.WidgetBox(
                "# Launcher",
                pn.Row(
                    #pn.Column(self.inferenceButton, self.cancelButton),
                    self.inferenceButton, self.cancelButton,
                    self.spinner, 
                    pn.Column(self.elapsedLabel, self.completionLabel)
                ),
                self.outputLog,
                sizing_mode='stretch_width',
            ),
            sizing_mode='stretch_both',
            min_height=300 # Forces the container to expand
        )
