import panel as pn
import param
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
#from directorySelect import DirectorySelect
from tkinter import Tk, filedialog
from directoryPicker import DirectoryPicker

class CommandRunner(param.Parameterized):
    command_input = param.String(default="")
    output_log = param.String(default="Terminal ready...")
    startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))

    def __init__(self, **params):
        super().__init__(**params)

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

        # 1. Text Input
        self.editor = pn.widgets.TextInput(
            name="Command input",
            placeholder="Type and press Enter...",
            sizing_mode='stretch_width',
            value=self.command_input
        )
        self.editor.link(self, value='command_input')
        self.editor.param.watch(self._execute, 'value')
        
        # 2. Button - Added 'visible=True' and explicit height to prevent 0px rendering
        self.run_btn = pn.widgets.Button(
            name="Run Command", 
            button_type="primary",
            height=40,
            min_width=120,
            visible=True,
            sizing_mode='fixed'
        )
        self.run_btn.on_click(self._execute)
        
        # 3. Output area
        self.console = pn.widgets.StaticText(
            name="Output log",
            value=f"<pre style='background:#f4f4f4; padding:5px;'>{self.output_log}</pre>",
            sizing_mode='stretch_width'
        )

    def _on_change(self, event):
        print("Selected:", event.new)
    
    def _execute(self, event):
        cmd = self.editor.value.strip()
        if not cmd: return

        try:
            startDateStr = self.startDate.strftime('%Y-%m-%d')
            endDateStr = self.endDate.strftime('%Y-%m-%d')
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            response = result.stdout if result.returncode == 0 else result.stderr
            self.output_log = response + " " + startDate + " " + endDate if response else "Done."
        except Exception as e:
            self.output_log = str(e)
            
        self.console.value = f"<pre style='background:#f4f4f4; padding:5px;'>{self.output_log}</pre>"

    def panel(self):
        return pn.Column(
            self.outputDirPicker.panel,
            pn.Row(self.startDatePicker, self.endDatePicker),
            "### CommandRunner",
            self.editor,
            self.run_btn, # Explicitly included in the column
            self.console,
            sizing_mode='stretch_width',
            min_height=300 # Forces the container to expand
        )

# --- DEBUG TEST ---
# Run this block directly to see if the button appears in isolation
#runner = CommandRunner()
#runner.panel().servable()
