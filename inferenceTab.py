import panel as pn
import param
import subprocess
import threading
import os

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
            options=['miles-credit', 'AIFS', 'AIFS Ensemble', 'Atlas', 'Aurora', 'DLWP', 'DLESyM', 'FourCastNet',
                    'FourCastNet 3', 'FengWu', 'FuXi', 'GraphCast', 'Pangu', 'SFNO', 'StormCast', 'StormScope',
                    'InterpModAFNO' 
            ],
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

        self.commandRunner = CommandRunner()

    def _on_run_click(self, event):
        """Wrapper to launch the execution in a thread."""
        #cmd = self.editor.value.strip()
        cmd = "credit_rollout_realtime -c model_predict_casper.yml"
        #cmd = "echo foo >> foo.txt"
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

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            response = result.stdout
        except Exception as e:
            response = f"Error: {str(e)}"

        self.output_log = response if response else "Done (no output)."
        self.spinner.value = False
        self.spinner.visible = False
        self.inferenceButton.disabled = False

    def panel(self):
        return pn.Column(
            pn.WidgetBox(
                '# Select Model',
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
            sizing_mode='stretch_width',
            min_height=300 # Forces the container to expand
        )
