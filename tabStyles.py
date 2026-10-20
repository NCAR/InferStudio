"""Tab-header styling shared by the app's main tabs (app_layout.py) and the
Inference tab's Weather/Climate sub-tabs (inference/inferenceTab.py), so
both tab bars look the same.

A tab's title is plain text, so a second line is a newline in the title.
pre-line wrapping displays it, and ::first-line keeps the title itself at
full size while the rest, e.g. "(experimental)", is a smaller note.
"""

TAB_HEADER_CSS = """
    .bk-tab {
        background: #f0f0f0; border-radius: 4px 4px 0 0;
        font-size: 13px; font-weight: 400; padding: 12px 24px;
        white-space: pre-line; text-align: center;
    }
    .bk-tab::first-line { font-size: 18px; font-weight: 600; }
    .bk-tab.bk-active { background: white; border-top: 3px solid #007bff; }
    .bk-tab.bk-active::first-line { font-weight: 700; }
    .bk-tabs-header { background: #e8e8e8; flex: 0 0 auto; }
"""


# A spinning ring after a tab's title while that tab is computing. Toggled
# by set_tab_busy, which adds a tab-busy-<index> class to the Tabs; the
# ring is CSS-animated in the browser, so it costs no traffic while it spins.
_MAX_BUSY_TABS = 8
BUSY_TAB_CSS = """
    @keyframes tab-busy-spin { to { transform: rotate(360deg); } }
""" + "".join(f"""
    :host(.tab-busy-{i}) .bk-tab:nth-child({i + 1})::after""" + ("," if i < _MAX_BUSY_TABS - 1 else "")
               for i in range(_MAX_BUSY_TABS)) + """ {
        content: ""; display: inline-block; vertical-align: middle;
        width: 14px; height: 14px; margin-left: 10px; box-sizing: border-box;
        border: 2.5px solid #c8d2dc; border-top-color: #007bff; border-radius: 50%;
        animation: tab-busy-spin 0.9s linear infinite;
    }
"""


def set_tab_busy(tabs, index, busy):
    """Show or hide the spinner on tab `index` of a pn.Tabs that has
    BUSY_TAB_CSS among its stylesheets."""
    cls = f"tab-busy-{index}"
    classes = [c for c in tabs.css_classes if c != cls] + ([cls] if busy else [])
    if classes != list(tabs.css_classes):
        tabs.css_classes = classes
