import panel as pn
import param
from datetime import datetime, timedelta

from tooltips import below_tooltip
from inference.earth2StudioRunner import CAMULATOR_START_TIMES, CAMULATOR_TRANSIENT_END

# CAMulator advances 6 hours per step and every step is saved.
STEP_HOURS = 6

RUN_LENGTH_UNITS = ['Days', 'Months', 'Years']

# Button label -> CAMulatorForcing mode.
FORCING_OPTIONS = {'Cyclic': 'cyclic', 'Transient': 'transient'}

FORCING_DESCRIPTION = (
    "Prescribed SST, sea ice, insolation and CO2. Cyclic repeats one "
    "climatological year, so a run can be any length. Transient follows "
    "the observed 1980-2014 record (a 9.7 GB download the first time), so "
    "a run has to end by 1 January 2015."
)


def _add_months(dt: datetime, months: int) -> datetime:
    """dt plus a whole number of months. CAMulator only starts on the 1st,
    so the day never has to be clamped."""
    years, month = divmod(dt.month - 1 + months, 12)
    return dt.replace(year=dt.year + years, month=month + 1)


class ClimateTimePicker(param.Parameterized):
    """Time settings for a climate run: a start date from the states the
    model ships with, a run length in days, months or years, and the
    forcing that drives it. Exposes the same start_time / end_date /
    timestep interface as TimePicker."""

    end_date = param.Date()

    def __init__(self, **params):
        super().__init__(**params)

        starts = sorted(CAMULATOR_START_TIMES)
        self.startDatePicker = pn.widgets.Select(
            name="Start Date",
            options={f"{t:%Y-%m-%d %H}Z": t for t in starts},
            value=datetime(2000, 1, 1),
            description=("CAMulator starts from the CAM states shipped with "
                         "it: 00Z on 1 January or 1 July, 1980-2014."),
        )

        self.runLengthLabel = pn.pane.Markdown("Run Length", margin=(0, 0, -5, 0))
        self.runLengthInput = pn.widgets.IntInput(
            value=1, start=1, end=100, width=100, margin=(0, 5, 5, 0))
        self.runLengthUnits = pn.widgets.RadioButtonGroup(
            options=RUN_LENGTH_UNITS, value='Months',
            button_type='primary', button_style='outline', margin=(0, 5, 5, 0))

        self.forcingLabel = pn.pane.Markdown("Forcing", margin=(0, 0, -5, 0))
        self.forcingButtons = pn.widgets.RadioButtonGroup(
            options=FORCING_OPTIONS, value='cyclic',
            button_type='primary', button_style='outline',
            description=below_tooltip(FORCING_DESCRIPTION),
            margin=(0, 5, 5, 0),
        )

        self._summary = pn.pane.Markdown("", margin=(5, 0, 0, 10))

        for widget in (self.startDatePicker, self.runLengthInput,
                       self.runLengthUnits, self.forcingButtons):
            widget.param.watch(self._update_end_date, 'value')

        self._update_end_date()

    @property
    def start_time(self):
        return self.startDatePicker.value

    @property
    def timestep(self) -> str:
        return f"{STEP_HOURS}h"

    @property
    def forcing_mode(self) -> str:
        return self.forcingButtons.value

    def _update_end_date(self, event=None):
        start = self.startDatePicker.value
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
        if self.forcing_mode == 'transient' and self.end_date > CAMULATOR_TRANSIENT_END:
            summary += (
                "<br><span style='color:#e74c3c'>Transient forcing ends at "
                f"{CAMULATOR_TRANSIENT_END:%Y-%m-%d %H}Z; shorten the run or "
                "use cyclic forcing.</span>"
            )
        self._summary.object = summary

    def panel(self):
        return pn.WidgetBox(
            "# Time Settings",
            self.startDatePicker,
            pn.Column(
                self.runLengthLabel,
                pn.Row(self.runLengthInput, self.runLengthUnits),
                margin=(0, 0, 0, 10),
                sizing_mode='stretch_width'
            ),
            pn.Column(
                self.forcingLabel,
                self.forcingButtons,
                margin=(0, 0, 0, 10),
                sizing_mode='stretch_width'
            ),
            self._summary,
        )
