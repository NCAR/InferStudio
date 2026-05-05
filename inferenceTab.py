import panel as pn
import param

from datetime import datetime, timedelta
from pathlib import Path

from directoryPicker import DirectoryPicker
from commandRunner import CommandRunner

class InferenceTab(param.Parameterized):
    startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))

    def __init__(self, **params):
        super().__init__(**params)

        self.commandRunner = CommandRunner()
        self.outputDirPicker = DirectoryPicker(start_path=Path.home())
        self.startDatePicker = pn.widgets.DatetimePicker(
            name="Start Date",
            value=self.startDate
        )
        self.startDatePicker.link(self, value='startDate')

        self.endDatePicker = pn.widgets.DatetimePicker(
            name="End Date",
            value=self.endDate
        )
        self.endDatePicker.link(self, value='endDate')


    def panel(self):
        return pn.Column(
            self.outputDirPicker.panel,
            pn.Row(self.startDatePicker, self.endDatePicker),
            self.commandRunner.panel(),
            #pn.Param(self.commandRunner, name="Command Runner"),
            sizing_mode='stretch_width',
            min_height=300 # Forces the container to expand
        )
