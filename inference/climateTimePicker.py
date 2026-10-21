import panel as pn
import param
from datetime import datetime, timedelta

from tooltips import below_tooltip
from inference.earth2StudioRunner import climate_start_times, run_end_limit

# The climate models all advance 6 hours per step, and every step is saved.
STEP_HOURS = 6

RUN_LENGTH_UNITS = ['Days', 'Months', 'Years']

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Where each model's start dates come from, shown under the start date.
START_NOTES = {
    'CAMulator': "CAMulator: the CAM states shipped with it, 1 January or 1 July, 1980-2014.",
    'DLESyMv0_ISCCP_ERA5': "DLESyM: ERA5, any month from February 1940 to December 2022.",
}

# Button label -> CAMulatorForcing mode.
FORCING_OPTIONS = {'Cyclic': 'cyclic', 'Transient': 'transient'}

FORCING_DESCRIPTION = (
    "CAMulator's prescribed SST, sea ice, insolation and CO2. Cyclic repeats one "
    "climatological year, so a run can be any length. Transient follows "
    "the observed 1980-2014 record (a 9.7 GB download the first time), so "
    "a run has to end by 1 January 2015."
)


def _add_months(dt: datetime, months: int) -> datetime:
    """dt plus a whole number of months. Climate runs only start on the
    1st, so the day never has to be clamped."""
    years, month = divmod(dt.month - 1 + months, 12)
    return dt.replace(year=dt.year + years, month=month + 1)


class ClimateTimePicker(param.Parameterized):
    """Time settings for a climate run: a start month the selected models
    can all start from, a run length in days, months or years, and
    CAMulator's forcing. Exposes the same start_time / end_date / timestep
    interface as TimePicker."""

    end_date = param.Date()

    def __init__(self, models=('CAMulator',), **params):
        super().__init__(**params)
        self._models = []
        self._starts = []
        self._setting = False

        self.startYearPicker = pn.widgets.Select(name="Start Year", width=100)
        self.startMonthPicker = pn.widgets.Select(name="Month", width=80)
        self.startNote = pn.pane.Markdown("", margin=(0, 10, 0, 10))

        self.runLengthLabel = pn.pane.Markdown("Run Length", margin=(0, 0, -5, 0))
        self.runLengthInput = pn.widgets.IntInput(
            value=1, start=1, end=100, width=100, margin=(0, 5, 5, 0))
        self.runLengthUnits = pn.widgets.RadioButtonGroup(
            options=RUN_LENGTH_UNITS, value='Months',
            button_type='primary', button_style='outline', margin=(0, 5, 5, 0))

        self.forcingLabel = pn.pane.Markdown("CAMulator Forcing", margin=(0, 0, -5, 0))
        self.forcingButtons = pn.widgets.RadioButtonGroup(
            options=FORCING_OPTIONS, value='cyclic',
            button_type='primary', button_style='outline',
            description=below_tooltip(FORCING_DESCRIPTION),
            margin=(0, 5, 5, 0),
        )
        self.forcingBox = pn.Column(
            self.forcingLabel,
            self.forcingButtons,
            margin=(0, 0, 0, 10),
            sizing_mode='stretch_width'
        )

        self._summary = pn.pane.Markdown("", margin=(5, 0, 0, 10))

        self.startYearPicker.param.watch(self._on_year, 'value')
        self.startMonthPicker.param.watch(self._on_month, 'value')
        for widget in (self.runLengthInput, self.runLengthUnits, self.forcingButtons):
            widget.param.watch(self._update_end_date, 'value')

        self.set_models(models, default=datetime(2000, 1, 1))

    def set_models(self, models, default=None):
        """Offer the start months every one of these models can start
        from, keeping the current one if it's still among them."""
        if not models:
            return      # nothing to fit; keep the dates as they are
        self._models = list(models)
        self._starts = sorted(set.intersection(
            *(set(climate_start_times(m)) for m in models)))
        self.forcingBox.visible = 'CAMulator' in models
        self.startNote.object = "<br>".join(
            START_NOTES[m] for m in models if m in START_NOTES)
        self._set_start(default or self.start_time)

    def _on_year(self, event):
        if not self._setting:
            # The same month if the new year has it, else its nearest.
            wanted = datetime(event.new, self.startMonthPicker.value or 1, 1)
            in_year = [t for t in self._starts if t.year == event.new]
            self._set_start(min(in_year, key=lambda t: abs(t - wanted)))

    def _on_month(self, event):
        if not self._setting:
            self._update_end_date()

    def _set_start(self, start):
        """Show the offered start nearest to `start`."""
        self._setting = True
        try:
            if self._starts:
                if start not in self._starts:
                    start = min(self._starts, key=lambda t: abs(t - (start or t)))
                self.startYearPicker.options = sorted({t.year for t in self._starts})
                self.startYearPicker.value = start.year
                self.startMonthPicker.options = {
                    MONTHS[t.month - 1]: t.month for t in self._starts if t.year == start.year}
                self.startMonthPicker.value = start.month
            else:
                self.startYearPicker.options = []
                self.startMonthPicker.options = []
        finally:
            self._setting = False
        self._update_end_date()

    @property
    def start_time(self):
        year, month = self.startYearPicker.value, self.startMonthPicker.value
        if year is None or month is None:
            return None
        return datetime(year, month, 1)

    @property
    def timestep(self) -> str:
        return f"{STEP_HOURS}h"

    @property
    def forcing_mode(self) -> str:
        return self.forcingButtons.value

    def _update_end_date(self, event=None):
        start = self.start_time
        if start is None:
            self._summary.object = (
                "<span style='color:#e74c3c'>These models share no start date; "
                "run them separately.</span>")
            return
        n = self.runLengthInput.value or 1
        unit = self.runLengthUnits.value
        if unit == 'Days':
            self.end_date = start + timedelta(days=n)
        else:
            self.end_date = _add_months(start, n * (12 if unit == 'Years' else 1))

        n_steps = int((self.end_date - start).total_seconds() // 3600) // STEP_HOURS
        summary = (
            f"**Steps:** {n_steps:,} (6-hourly, every step saved) &nbsp;&nbsp; "
            f"**End Date:** {self.end_date:%Y-%m-%d %H:%M:%S}"
        )
        for model in self._models:
            limit = run_end_limit(model, self.forcing_mode)
            if limit is not None and self.end_date > limit:
                summary += (
                    "<br><span style='color:#e74c3c'>Transient forcing ends at "
                    f"{limit:%Y-%m-%d %H}Z; shorten the run or use cyclic "
                    "forcing.</span>"
                )
        self._summary.object = summary

    def panel(self):
        return pn.WidgetBox(
            "### Time Settings",
            pn.Row(self.startYearPicker, self.startMonthPicker,
                   pn.pane.Markdown("00Z on the 1st", margin=(30, 5, 0, 5)),
                   margin=(0, 0, 0, 0)),
            self.startNote,
            pn.Column(
                self.runLengthLabel,
                pn.Row(self.runLengthInput, self.runLengthUnits),
                margin=(0, 0, 0, 10),
                sizing_mode='stretch_width'
            ),
            self.forcingBox,
            self._summary,
            sizing_mode='stretch_width',
        )
