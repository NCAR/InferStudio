import pandas as pd
import panel as pn
import param

# Initialize the extension (standard for Jupyter/Casper environments)
pn.extension()

class DatasetBrowser(param.Parameterized):
    checked_items = param.List(default=[])
    active_dataset = param.String(default="")

    def __init__(self, datasets, **params):
        super().__init__(**params)
        self.datasets = list(datasets)
        self._rows = {}
        self._checkboxes = {}   # name -> Checkbox widget, for two-way sync
        self._syncing = False   # guards against recursive updates while
                                 # programmatically setting checkbox values

        # Build once, keep a handle so we can append to it later
        self._column = pn.Column(
            *[self._make_row(d) for d in self.datasets],
            sizing_mode='stretch_width',
            max_height=600,
            scroll=True,
            margin=(0, 10, 0, 0),
            styles={'border': '1px solid #ddd', 'border-radius': '4px', 'background': 'white'}
        )

    def _get_row_style(self, name):
        """Calculates the CSS for a row based on whether it is active."""
        is_active = (name == self.active_dataset)
        return {
            'background': '#e8f2ff' if is_active else 'transparent',
            'border-left': '5px solid #007bff' if is_active else '5px solid transparent',
            'display': 'flex',
            'align-items': 'center',
            'width': '100%',
            'border-radius': '0 4px 4px 0',
            'transition': 'background 0.2s', # Smooth color transition
            'padding' : '0px 2px',
            'margin' : '0pz'
        }

    def _set_active(self, name):
        """Centralized logic to move the highlight and refresh the UI."""
        self.active_dataset = name
        self._update_ui()

    def _make_row(self, name):
        # 1. Checkbox — only one may be checked at a time (see update_checked)
        cb = pn.widgets.Checkbox(
            name="", value=False, width=10, 
            align='center', margin=(10, 0, 2, 0) 
        )
        self._checkboxes[name] = cb

        def update_checked(event):
            if self._syncing:
                # This value change came from _sync_checkbox_widgets reacting
                # to an external checked_items change, not a user click —
                # don't re-derive checked_items from it (would be redundant
                # and risks feedback loops).
                return

            if event.new:
                # Enforce single-selection: checking this box replaces
                # whatever was checked before, rather than adding to it.
                self._set_active(name)
                self.checked_items = [name]
            else:
                current = list(self.checked_items)
                if name in current:
                    current.remove(name)
                self.checked_items = current

        cb.param.watch(update_checked, 'value')

        # 2. Button styled as a flat label for single-select/highlight
        # align='center' and line-height: 30px match the checkbox height
        btn = pn.widgets.Button(
            name=name, 
            button_type='default', 
            sizing_mode='stretch_width',
            align='center',
            stylesheets=["""
                .bk-btn {
                    background: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    text-align: left !important;
                    padding-left: 5px !important;
                    line-height: 6px !important;
                    font-size: 14px !important;
                    color: #333 !important;
                    cursor: pointer;
                    outline: none !important;
                    box-shadow: none !important;
                }
                .bk-btn:hover { 
                    background: rgba(0,0,0,0.05) !important; 
                }
            """]
        )
        
        # Highlight row when the dataset name is clicked
        btn.on_click(lambda event: self._set_active(name))

        # 3. Create the Row container
        row = pn.Row(
            cb, 
            btn, 
            styles=self._get_row_style(name), 
            sizing_mode='stretch_width',
            margin=0
        )
        
        self._rows[name] = row
        return row

    def _update_ui(self):
        """Forces the CSS styles of all rows to refresh."""
        for name, row in self._rows.items():
            row.styles = self._get_row_style(name)
            # Explicitly trigger the parameter update for older Panel versions
            row.param.trigger('styles')

    @param.depends('checked_items', watch=True)
    def _sync_checkbox_widgets(self):
        """Keep every row's Checkbox visually in sync with checked_items,
        including when it's set programmatically from outside this class
        (e.g. app_layout.py selecting a newly-completed simulation)."""
        checked_set = set(self.checked_items)
        self._syncing = True
        try:
            for name, cb in self._checkboxes.items():
                desired = name in checked_set
                if cb.value != desired:
                    cb.value = desired
        finally:
            self._syncing = False

    def add_datasets(self, names):
        """Append new dataset rows in place without disturbing existing ones."""
        for name in names:
            if name in self._rows:
                continue
            self.datasets.append(name)
            self._column.append(self._make_row(name))

    @property
    def panel(self):
        return self._column
