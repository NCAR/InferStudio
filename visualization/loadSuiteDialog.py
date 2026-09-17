import os
from pathlib import Path

import panel as pn

# Directory names recognized as supported model-output subdirectories,
# mirroring inference/inferenceTab.py's MILES_CREDIT_MODEL_LIST +
# EARTH2STUDIO_MODEL_LIST. Duplicated rather than imported -- like
# datasetPlot.py's EARTH2STUDIO_FORMAT_MODELS, this is only a UI hint for
# the directory browser, not the source of truth for what's runnable.
SUPPORTED_MODEL_DIRS = frozenset({"WXFormer", "AIFS", "Aurora", "Pangu", "FourCastNet3"})

# Deliberately darker than the ✅/⚠️ emoji glyphs' own bright green/yellow,
# so each icon keeps a visible edge against the button's fill rather than
# blending into a same-toned background.
_VALID_SUITE_COLOR = "#1b5e20"    # dark green
_INVALID_SUITE_COLOR = "#8a6d00"  # dark amber


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

        self.currentPathDisplay = pn.widgets.TextInput(
            value=self.current_path_val,
            disabled=True,
            sizing_mode="stretch_width",
        )

        # Re-lists the currently viewed directory without navigating away
        # from it -- useful for a suite that's still being written by a
        # running inference job, where a subdirectory (e.g. a model that
        # just started) wouldn't otherwise show up until the dialog is
        # reopened.
        self.refresh_button = pn.widgets.Button(
            name="🔄",
            width=40,
            margin=(5, 0, 5, 5),
        )
        self.refresh_button.on_click(lambda e: self._refresh())

        self.list_container = pn.Column(
            height=350,
            scroll=True,
            styles={"border": "1px solid #ccc", "background": "white"},
        )

        # Name and background are both kept in sync with the valid/invalid
        # check in _refresh() -- see there for the actual values.
        self.select_button = pn.widgets.Button(
            name="Confirm Selection",
            sizing_mode="stretch_width",
        )
        self.select_button.on_click(self._select)

        # Set by _refresh() to tell the user, below the button they'd click
        # to proceed, whether the currently viewed directory's contents are
        # entirely recognized model subdirectories or not.
        self.suite_indicator = pn.pane.Markdown("", margin=(5, 0, 0, 0))

        # Inline status/error message shown inside the dialog itself, so
        # a failed scan (e.g. no supported model output found) is visible
        # right where the user is working, in addition to (not instead
        # of) any pn.state.notifications toast the caller may also show.
        self.status = pn.pane.Markdown("", margin=(5, 0, 0, 0))

        self.dialog = pn.Column(
            "### 📁 Select the root directory of a previously-run simulation suite.<br><br>"
            "This is the top-level folder that contains one "
            "subdirectory per AI model that was part of the suite "
            "(e.g. AIFS, Aurora, WXFormer).<br>"
            "By default, it will be in your scratch directory on glade, and be named something "
            "like InferStudio_Aurora_Pangu_2026_08_28_11:19:57",
            pn.Row(self.currentPathDisplay, self.refresh_button, sizing_mode="stretch_width"),
            self.list_container,
            self.select_button,
            self.suite_indicator,
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

        self._refresh()

    def _refresh(self):
        self.currentPathDisplay.value = self.current_path_val

        try:
            entries = os.listdir(self.current_path_val)
            dirs = sorted(
                (d for d in entries
                 if os.path.isdir(os.path.join(self.current_path_val, d))),
                key=str.lower,
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

            # Only every entry (directories AND stray files alike) being a
            # recognized model name counts as a match -- anything else in
            # the directory means it isn't the suite root, even if it also
            # happens to contain a real model subdirectory.
            if entries and all(e in SUPPORTED_MODEL_DIRS for e in entries):
                self.select_button.name = "Confirm Selection ✅"
                self.select_button.styles = {
                    "background": _VALID_SUITE_COLOR, "color": "white"}
                self.suite_indicator.object = (
                    "✅ **This looks like a valid simulation suite.**"
                )
            else:
                self.select_button.name = "Confirm Selection ⚠️"
                self.select_button.styles = {
                    "background": _INVALID_SUITE_COLOR, "color": "white"}
                self.suite_indicator.object = (
                    "⚠️ **This does not look like a valid simulation suite.**"
                )

        except Exception as e:
            self.list_container.objects = [
                pn.pane.Markdown(f"**Error:** {e}")
            ]
            self.select_button.name = "Confirm Selection ⚠️"
            self.select_button.styles = {
                "background": _INVALID_SUITE_COLOR, "color": "white"}
            self.suite_indicator.object = ""

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
        # Scanning a full suite can take a moment; disabling and relabeling
        # the button both prevents a double-submit and makes clear that the
        # click registered while the caller does that work (report_error /
        # close below are what eventually restore or dismiss this).
        self.select_button.disabled = True
        self.select_button.name = "Loading..."
        self._on_select_callback(self.current_path_val)

    def report_error(self, message: str):
        """Show an inline error in the dialog (called by the caller after
        a failed scan) without closing the modal, so the user can
        navigate to a different directory and try again."""
        self.status.object = f"**Error:** {message}"
        self.select_button.disabled = False
        # The directory itself hasn't changed, so this just restores the
        # button's icon/color/label from "Loading..." back to whatever
        # _select overwrote it from.
        self._refresh()

    def close(self):
        """Close the modal (called by the caller after a successful
        scan)."""
        self.select_button.disabled = False
        self._refresh()
        self.modal.hide()
