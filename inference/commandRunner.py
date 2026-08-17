import panel as pn
import param
import subprocess
import threading
import os

from pathlib import Path
from datetime import datetime, timedelta

class CommandRunner(param.Parameterized):
    command_input = param.String(default="")
    output_log = param.String(default="Terminal ready...")

    def __init__(self, **params):
        super().__init__(**params)

        self.editor = pn.widgets.TextInput(
            name="Command input",
            placeholder="credit_rollout_realtime -c model_predict_casper.yml",
            sizing_mode='stretch_width',
            value=self.command_input
        )
        self.editor.link(self, value='command_input')
        self.editor.param.watch(self._execute, 'value')
        
        self.run_btn = pn.widgets.Button(
            name="Run Command", 
            button_type="primary",
            height=40,
            min_width=120,
            visible=True,
            sizing_mode='fixed'
        )
        self.run_btn.on_click(self._on_run_click)

        self.clear_btn = pn.widgets.Button(
            name="Clear Output", 
            button_type="default",
            height=40,
            width=120
        )
        self.clear_btn.on_click(self._clear_output)

        self.spinner = pn.indicators.LoadingSpinner(
            width=30, height=30, value=False, color="primary", visible=False
        )
        
        self.console = pn.widgets.StaticText(
            name="Output log",
            value=f"<pre style='background:#f4f4f4; padding:5px;'>{self.output_log}</pre>",
            sizing_mode='stretch_width'
        )

    def _on_change(self, event):
        print("Selected:", event.new)

    def _clear_output(self, event):
        self.console.value = "<pre style='background:#f4f4f4; padding:5px;'>Terminal cleared...</pre>"

    def _on_run_click(self, event):
        """Wrapper to launch the execution in a thread."""
        cmd = self.editor.value.strip()
        if not cmd: return

        # UI updates happen immediately here
        self.spinner.value = True
        self.spinner.visible = True
        self.run_btn.disabled = True
        self.console.value = "<pre style='color: blue;'>Running command...</pre>"

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
        self.run_btn.disabled = False
        self.console.value = f"<pre style='background:#f4f4f4; padding:5px; white-space: pre-wrap;'>{self.output_log}</pre>"

    def panel(self):
        return pn.Column(
            pn.Card(
                self.editor,
                pn.Row(self.run_btn, self.clear_btn, self.spinner, align='start'),
                self.console,
                title="Command Runner",
                collapsed=True
            ),
            sizing_mode='stretch_width',
            min_height=300 # Forces the container to expand
        )

# --- DEBUG TEST ---
# Run this block directly to see if the button appears in isolation
#runner = CommandRunner()
#runner.panel().servable()
