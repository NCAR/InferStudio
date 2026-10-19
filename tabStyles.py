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
