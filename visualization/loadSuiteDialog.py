import os
from html import escape
from pathlib import Path

import panel as pn

# Directory names recognized as supported model-output subdirectories,
# mirroring inference/inferenceTab.py's MILES_CREDIT_MODEL_LIST +
# EARTH2STUDIO_MODEL_LIST. Duplicated rather than imported -- like
# datasetPlot.py's EARTH2STUDIO_FORMAT_MODELS, this is only a UI hint for
# the directory browser, not the source of truth for what's runnable.
SUPPORTED_MODEL_DIRS = frozenset({"WXFormer", "AIFS", "Aurora", "Pangu", "FourCastNet3"})

# Banner colours per verdict: (border/title, background). Saturated border
# and title on a pale fill, so the verdict reads from across the room
# without the body text losing contrast.
_BANNER_COLORS = {
    "suite": ("#1b5e20", "#e3f5e4"),    # green
    "invalid": ("#b71c1c", "#fdecea"),  # red
}

# Bigger, bolder label for the Confirm button - it carries the verdict too.
# Disabled (on anything but a suite) it stays full-strength red rather
# than fading to a washed-out pink.
_CONFIRM_STYLESHEET = """
.bk-btn { font-size: 16px; font-weight: 700; }
.bk-btn:disabled { opacity: 1; cursor: not-allowed; }
"""

_FOLDER_BUTTON_STYLES = {
    "justify-content": "flex-start",
    "text-align": "left",
    "display": "flex",
    "width": "100%",
}


def _visible_entries(path):
    """Directory entries, ignoring hidden ones (.ipynb_checkpoints, a
    diff's .tmp file, ...) - those never make a folder more or less of a
    suite."""
    return [e for e in os.listdir(path) if not e.startswith(".")]


def _is_suite(path):
    """True if every visible entry in `path` is a supported model
    directory (and there is at least one). Anything else in the directory
    means it isn't the suite root, even if it also happens to contain a
    real model subdirectory."""
    try:
        entries = _visible_entries(path)
    except OSError:
        return False
    return bool(entries) and all(
        e in SUPPORTED_MODEL_DIRS and os.path.isdir(os.path.join(path, e))
        for e in entries)


def _nearest_existing_dir(path):
    """`path` itself if it's a directory, else its closest ancestor that is."""
    p = Path(path)
    while not p.is_dir() and p != p.parent:
        p = p.parent
    return str(p)


def _classify(path):
    """Verdict on `path` as a suite root: (kind, title, detail HTML), where
    kind is "suite" or "invalid"."""
    if not os.path.exists(path):
        return ("invalid", "No such directory",
                f"<code>{escape(path)}</code> does not exist.")
    if not os.path.isdir(path):
        return ("invalid", "Not a directory",
                f"<code>{escape(path)}</code> is a file. Choose the suite's folder.")
    try:
        entries = _visible_entries(path)
    except OSError as e:
        return ("invalid", "Can't read this directory", escape(str(e)))

    if _is_suite(path):
        models = ", ".join(sorted(entries, key=str.lower))
        return ("suite", "Valid simulation suite",
                f"Models: <b>{escape(models)}</b>")

    name = os.path.basename(path)
    if name in SUPPORTED_MODEL_DIRS and _is_suite(os.path.dirname(path)):
        return ("invalid", "Not a simulation suite",
                f"This is the <b>{escape(name)}</b> folder <i>inside</i> a suite. "
                "Go up one level (<b>..</b>) and load the suite from there.")

    suites_below = [e for e in entries if _is_suite(os.path.join(path, e))]
    if suites_below:
        n = len(suites_below)
        return ("invalid", "Not a simulation suite",
                f"But {n} folder{'s' if n > 1 else ''} below "
                f"{'are suites' if n > 1 else 'is a suite'}, marked ✅. "
                "Open one to load it.")

    models = [e for e in entries if e in SUPPORTED_MODEL_DIRS]
    if models:
        others = sorted((e for e in entries if e not in SUPPORTED_MODEL_DIRS),
                        key=str.lower)
        shown = ", ".join(others[:5]) + (", …" if len(others) > 5 else "")
        return ("invalid", "Not a simulation suite",
                f"It has model folders ({escape(', '.join(sorted(models)))}), "
                f"but also other things that a suite never contains: {escape(shown)}.")

    return ("invalid", "Not a simulation suite",
            "A suite folder contains only model folders "
            f"({escape(', '.join(sorted(SUPPORTED_MODEL_DIRS)))}), "
            "and this one has none.")


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

    The path box is editable: typing a path and pressing Enter goes
    straight there. Whatever directory is showing gets a verdict banner:
    green for something that looks like a suite, red (with the reason)
    for anything else. The Confirm button can only be clicked on a green
    one.

    `on_select` is called with the chosen path (a string) once the user
    clicks "Confirm Selection". It's the caller's responsibility to
    actually scan that directory and report success/failure back via
    `report_error` / `close`.
    """

    def __init__(self, start_path, on_select, width=400):
        self.current_path_val = str(Path(start_path).expanduser().resolve())
        self._on_select_callback = on_select
        self._is_valid = False

        # Editable: fires on Enter (or leaving the box), not per keystroke.
        self.currentPathDisplay = pn.widgets.TextInput(
            value=self.current_path_val,
            placeholder="Type a path and press Enter",
            sizing_mode="stretch_width",
        )
        self.currentPathDisplay.param.watch(self._on_path_typed, "value")

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
            height=300,
            scroll=True,
            styles={"border": "1px solid #ccc", "background": "white"},
        )

        # The verdict on the directory being viewed - see _refresh().
        self.suite_indicator = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=(10, 5, 5, 5))

        # Name, colour and enabled state all follow the verdict in _refresh().
        self.select_button = pn.widgets.Button(
            name="Confirm Selection",
            sizing_mode="stretch_width",
            height=48,
            stylesheets=[_CONFIRM_STYLESHEET],
        )
        self.select_button.on_click(self._select)

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
            "like InferStudio_Aurora_Pangu_2026_08_28_11:19:57<br>"
            "Browse below, or type a path and press Enter.",
            pn.Row(self.currentPathDisplay, self.refresh_button, sizing_mode="stretch_width"),
            self.list_container,
            self.suite_indicator,
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

        self._refresh()

    def _folder_button(self, label, target, suite=False):
        btn = pn.widgets.Button(
            name=label,
            sizing_mode="stretch_width",
            styles=_FOLDER_BUTTON_STYLES,
            # Suites stand out in the listing, so they can be spotted
            # without opening every folder.
            button_type="success" if suite else "default",
            button_style="outline" if suite else "solid",
        )
        btn.on_click(lambda e: self._go_to(target))
        return btn

    def _refresh(self):
        path = self.current_path_val
        # Assigning the same value doesn't re-fire _on_path_typed.
        self.currentPathDisplay.value = path

        if os.path.isdir(path):
            try:
                dirs = sorted(
                    (d for d in _visible_entries(path)
                     if os.path.isdir(os.path.join(path, d))),
                    key=str.lower,
                )
                buttons = [self._folder_button("📁 ..", os.path.dirname(path))]
                for d in dirs:
                    full = os.path.join(path, d)
                    suite = _is_suite(full)
                    buttons.append(self._folder_button(
                        f"{'✅' if suite else '📁'} {d}", full, suite))
                self.list_container.objects = buttons
            except Exception as e:
                self.list_container.objects = [
                    pn.pane.Markdown(f"**Error:** {e}")
                ]
        else:
            # A typed path that doesn't exist: offer the way back to the
            # closest folder that does.
            back = _nearest_existing_dir(path)
            self.list_container.objects = [
                self._folder_button(f"↩ Back to {back}", back)
            ]

        kind, title, detail = _classify(path)
        self._is_valid = kind == "suite"
        border, background = _BANNER_COLORS[kind]
        icon = "✅" if self._is_valid else "❌"
        self.suite_indicator.object = (
            f"<div style='border:3px solid {border}; background:{background};"
            f" border-radius:6px; padding:10px 14px; color:#222;'>"
            f"<div style='color:{border}; font-size:20px; font-weight:800;"
            f" text-transform:uppercase; letter-spacing:0.02em;'>"
            f"{icon} {escape(title)}</div>"
            f"<div style='font-size:14px; margin-top:6px;'>{detail}</div>"
            f"</div>"
        )

        if self._is_valid:
            self.select_button.name = "✅ Load this suite"
            self.select_button.button_type = "success"
            self.select_button.disabled = False
        else:
            self.select_button.name = "❌ Not a suite, can't load"
            self.select_button.button_type = "danger"
            self.select_button.disabled = True

    def _on_path_typed(self, event):
        typed = (event.new or "").strip()
        if not typed:
            # Emptied box: put the current path back.
            self._refresh()
            return
        path = os.path.expandvars(os.path.expanduser(typed))
        if not os.path.isabs(path):
            path = os.path.join(self.current_path_val, path)
        if os.path.normpath(path) != self.current_path_val:
            self._go_to(path)

    def _go_to(self, path):
        # Navigating clears any previous error, since the user is
        # actively looking for a different directory now.
        self.status.object = ""
        self.current_path_val = os.path.normpath(path)
        self._refresh()

    def _select(self, _):
        if not self._is_valid:
            return
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
        # The directory itself hasn't changed, so this just restores the
        # button's icon/color/label/enabled state from "Loading..." back
        # to whatever _select overwrote it from.
        self._refresh()

    def close(self):
        """Close the modal (called by the caller after a successful
        scan)."""
        self._refresh()
        self.modal.hide()
