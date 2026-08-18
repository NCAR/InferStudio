import os
from pathlib import Path

import panel as pn


class LoadSuiteDialog:
    """A 'Load Existing Suite' button + directory-browser modal for the
    Visualization tab, letting a user navigate to and select a
    previously-run simulation suite directory (e.g. from an earlier
    InferStudio session) rather than only seeing suites produced in the
    current session.

    Mirrors the directory-browsing modal pattern already used in
    OutputParams (inference/outputParams.py) for choosing where to WRITE
    new output — this is a separate, standalone implementation for
    choosing an EXISTING directory to READ from, so as not to risk
    modifying that already-working file.

    `on_select` is called with the chosen path (a string) once the user
    clicks "Confirm Selection". It's the caller's responsibility to
    actually scan that directory and report success/failure back via
    `report_error` / `close`.
    """

    def __init__(self, start_path, on_select, width=400):
        self.current_path_val = str(Path(start_path).expanduser().resolve())
        self._on_select_callback = on_select
        self._instruction_notification = None

        self.currentPathDisplay = pn.widgets.TextInput(
            value=self.current_path_val,
            disabled=True,
            sizing_mode="stretch_width",
        )

        self.list_container = pn.Column(
            height=350,
            scroll=True,
            styles={"border": "1px solid #ccc", "background": "white"},
        )

        self.select_button = pn.widgets.Button(
            name="Confirm Selection ✅",
            button_type="success",
            sizing_mode="stretch_width",
        )
        self.select_button.on_click(self._select)

        # Inline status/error message shown inside the dialog itself, so
        # a failed scan (e.g. no supported model output found) is visible
        # right where the user is working, in addition to (not instead
        # of) any pn.state.notifications toast the caller may also show.
        self.status = pn.pane.Markdown("", margin=(5, 0, 0, 0))

        self.dialog = pn.Column(
            "### 📁 Select a Simulation Suite Directory",
            self.currentPathDisplay,
            self.list_container,
            self.select_button,
            self.status,
            width=width,
        )

        self.modal = pn.Modal(self.dialog, name="Load Existing Suite", margin=0)
        self.open_button = self.modal.create_button(
            "toggle",
            name="Load Existing Suite",
            button_type="primary",
            sizing_mode="stretch_width",
        )

        # Show an instructional notification whenever the modal opens
        # (not just once at app launch), explaining what kind of
        # directory to pick. Dismissible manually by the user (a normal
        # duration=0 notification always is), and also dismissed
        # automatically the moment "Confirm Selection" is clicked — see
        # _select below.
        self.modal.param.watch(self._on_modal_toggle, 'open')

        self._refresh()

    def _on_modal_toggle(self, event):
        if event.new:  # modal just opened
            if pn.state.notifications:
                self._instruction_notification = pn.state.notifications.info(
                    "Select the root directory of a previously-run simulation "
                    "suite.<br><br>"
                    "This is the top-level folder that contains one "
                    "subdirectory per AI model that was part of the suite "
                    "(e.g. AIFS, Aurora, WXFormer).",
                    duration=0,
                )
        else:
            # Modal closed some other way (e.g. the dialog's own X
            # button) without confirming a selection — clear our
            # reference so we don't try to destroy an already-gone
            # notification later.
            self._instruction_notification = None

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
        # Navigating clears any previous error, since the user is
        # actively looking for a different directory now.
        self.status.object = ""

        if folder == "..":
            new_path = os.path.dirname(self.current_path_val)
        else:
            new_path = os.path.join(self.current_path_val, folder)

        new_path = os.path.normpath(new_path)

        if os.path.isdir(new_path):
            self.current_path_val = new_path
            self._refresh()

    def _select(self, _):
        # Dismiss the instructional notification the moment the user
        # confirms a selection, regardless of whether that selection
        # turns out to be valid.
        if self._instruction_notification is not None:
            try:
                self._instruction_notification.destroy()
            except Exception:
                pass  # already dismissed manually by the user — fine
            self._instruction_notification = None

        self._on_select_callback(self.current_path_val)

    def report_error(self, message: str):
        """Show an inline error in the dialog (called by the caller after
        a failed scan) without closing the modal, so the user can
        navigate to a different directory and try again."""
        self.status.object = f"**Error:** {message}"

    def close(self):
        """Close the modal (called by the caller after a successful
        scan)."""
        self.modal.hide()
