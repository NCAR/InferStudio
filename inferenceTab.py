import panel as pn
import param
import subprocess
import threading
import os
import shutil

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

            #options={'1 hour':1, '6 hour':6, '12 hour':12, '24 hour':24},
        self.modelPicker = pn.widgets.RadioButtonGroup(
            name="Select Model",
            options=['WXFormer', 'AIFS', 'Aurora', 'FourCastNet3', 'GraphCast', 'Pangu', 'SFNO'],
            button_type='primary',
            button_style='outline',
            margin=(0,5,5,0)
        )

        #self.simulationNamePicker = pn.widgets.TextInput(name='Simulation Name', placeholder='Enter a name for your simulation')

        self.outputDirPicker = DirectoryPicker(start_path=Path.home())

        self.timePicker = TimePicker()

        self.inferenceButton = pn.widgets.Button(
            name="Run Inference",
            button_type="primary",
        )
        self.inferenceButton.on_click(self._on_run_click)

        self.spinner = pn.indicators.LoadingSpinner(
            width=30, height=30, value=False, color="primary", visible=False
        )

        self.outputLog = pn.widgets.TextAreaInput(
            name="Output Log",
            value="",
            height=200,
            sizing_mode="stretch_width",
        )

        self.commandRunner = CommandRunner()

    def _replaceParams(self):
        #start = self.timePicker.startDate
        #end = self.timePicker.endDate
        #inc = self.timePicker.increment

        print("CPV " + self.outputDirPicker.current_path_val)

        with open('model_predict_casper.yml', 'r') as f:
            content = f.read()

        #forecast_start = start
        #forecast_end   = end

        content = content.replace(
            'forecast_start_time: "2025-07-02 00:00:00"',
            f'forecast_start_time: "{self.timePicker.startDatePicker.value}"'
            #f'forecast_start_time: "{forecast_start}"'
        )
        content = content.replace(
            'forecast_end_time: "2025-07-02 02:00:00"',
            f'forecast_end_time: "{self.timePicker.endDatePicker.value}"'
            #f'forecast_end_time: "{forecast_end}"'
        )
        content = content.replace(
            'forecast_timestep: "1h"',
            #f'forecast_timestep: "{inc}"'
            #f'forecast_timestep: "{self.timePicker.increment}"'
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
        """Wrapper to launch the execution in a thread."""
        #cmd = self.editor.value.strip()
        #cmd = "credit_rollout_realtime -c model_predict_casper.yml"
        #cmd = "echo foo >> foo.txt"
        print("hmmm should do it ...")
        self._replaceParams()
        #cmd = "cat model_predict_casper.yml > new.yml"
        #outFile = self.outputDirPicker.current_path_val + '/' + self.outputDirPicker.simulationNamePicker.value + '.yml'
        #cmd += f" && echo {outFile} >> new.yml"

        cmd = f"""python /glade/work/pearse/credit-panel/miles-credit/applications/gfs_init.py -c {self.configFile} &&
                 python /glade/work/pearse/credit-panel/miles-credit/applications/rollout_realtime.py -c {self.configFile}"""
        #cmd = f"echo '{myCmd}' >> '{self.configFile}'"

        if not cmd: return

        # UI updates happen immediately here
        self.spinner.value = True
        self.spinner.visible = True
        self.inferenceButton.disabled = True

        # Launch the actual subprocess in the background
        thread = threading.Thread(target=self._execute, args=(cmd,))
        thread.start()

    def _execute(self, cmd):
        """The actual heavy lifting, now running in a background thread."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
    
        self.outputLog.value = ""  # clear previous output
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            for line in process.stdout:
                self.outputLog.value += line
            process.wait()
            if process.returncode != 0:
                self.outputLog.value += f"\nProcess exited with code {process.returncode}"
        except Exception as e:
            self.outputLog.value = f"Error: {str(e)}"
        finally:
            self.spinner.value = False
            self.spinner.visible = False
            self.inferenceButton.disabled = False
    
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
                pn.Row(self.inferenceButton, self.spinner),
                sizing_mode='stretch_width',
            ),
            #self.commandRunner.panel(),
            self.outputLog,
            sizing_mode='stretch_width',
            min_height=300 # Forces the container to expand
        )
