"""Widget tooltips that open BELOW the widget, wrapped to a readable width.

Passing a plain string as a Panel widget's `description` gets a tooltip
hard-wired to the widget's right (panel.widgets.base builds it with
position='right' and a 300px max width). In the sidebar and the Inference
tab's button rows that covers the neighbouring controls. Handing Panel a
ready-made bokeh Tooltip instead is supported, and lets the placement and
width be set here:

    pn.widgets.Button(name="AIFS", description=below_tooltip("..."))
"""

from html import escape

from bokeh.models import Tooltip
from bokeh.models.dom import HTML

# Approximate characters per line. `ch` is the width of a "0" in the
# tooltip's own font, so 40ch is close to 40 characters of ordinary text.
TOOLTIP_LINE_CHARS = 40

# Bokeh's default tooltip is near-white on white with a pale grey border and
# is easy to miss. These match the app header and notification toast
# (#091422, see app_layout.py and static/styles.css) so the tip reads as
# part of the app. The --tooltip-* variables are Bokeh's own; the arrow
# takes --tooltip-arrow-color, so it has to be set to match too.
_TOOLTIP_CSS = (
    ":host {"
    " --tooltip-color: #091422; --tooltip-text: #ffffff;"
    " --tooltip-border: #091422; --tooltip-arrow-color: #091422;"
    " --tooltip-arrow-height: 8px; --tooltip-arrow-half-width: 8px;"
    " opacity: 1; font-size: 13px; line-height: 1.4;"
    " padding: 8px 12px; border-radius: 6px;"
    " box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);"
    " white-space: normal; width: max-content;"
    f" max-width: {TOOLTIP_LINE_CHARS}ch;"
    " }"
)


def below_tooltip(text):
    """A Tooltip for a widget's `description`, opening beneath it.

    Anchored to the widget's bottom edge, centred, with the arrow pointing
    up at it. width:max-content keeps a short tip on one line rather than
    stretching it to the full max width.
    """
    return Tooltip(
        content=HTML(escape(text)),
        position="bottom_center",
        attachment="below",
        stylesheets=[_TOOLTIP_CSS],
        syncable=False,
    )
