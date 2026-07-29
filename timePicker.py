import panel as pn
import param
from datetime import datetime, timedelta

VALID_CYCLES = {0, 6, 12, 18}

# Check available dates for gfs_init.py with
# gsutil ls gs://global-forecast-system/gdas.20260610/

STEP_HOURS = {'1h': 1, '6h': 6, '12h': 12, '24h': 24}


class TimePicker(param.Parameterized):
    # Computed automatically from startDatePicker + leadTimeSlider +
    # incrementButtons — not directly user-editable anymore.
    end_date = param.Date()

    def __init__(self, **params):
        super().__init__(**params)

        self.startDatePicker = pn.widgets.DatetimePicker(
            name="Start Date",
            value=self._snap_to_cycle(datetime.now() - timedelta(hours=24))
        )
        self.startDatePicker.param.watch(self._snap_start, 'value')

        self.incrementLabel = pn.pane.Markdown("Time Step Increment", margin=(0, 0, -5, 0))
        self.incrementButtons = pn.widgets.RadioButtonGroup(
            name="",
            options={'1 hour': '1h', '6 hour': '6h', '12 hour': '12h', '24 hour': '24h'},
            value='6h',
            button_type='primary',
            button_style='outline',
            margin=(0, 5, 5, 0)
        )
        self.incrementButtonsGroup = pn.Row(self.incrementButtons)

        self.leadTimeLabel = pn.pane.Markdown("Lead Time (steps)", margin=(0, 0, -5, 0))
        self.leadTimeSlider = pn.widgets.IntSlider(
            name="",
            start=1,
            end=40,
            value=3,
            step=1,
            margin=(0, 5, 5, 0),
        )

        self._summary = pn.pane.Markdown("", margin=(5, 0, 0, 10))

        # Recompute end_date whenever any of the three inputs change
        self.startDatePicker.param.watch(self._update_end_date, 'value')
        self.incrementButtons.param.watch(self._update_end_date, 'value')
        self.leadTimeSlider.param.watch(self._update_end_date, 'value_throttled')

        self._update_end_date()

    @staticmethod
    def _snap_to_cycle(dt: datetime) -> datetime:
        snapped_hour = (dt.hour // 6) * 6
        return dt.replace(hour=snapped_hour, minute=0, second=0, microsecond=0)

    def _snap_start(self, event):
        snapped = self._snap_to_cycle(event.new)
        if snapped != event.new:
            # Setting .value here will re-trigger this watcher via the
            # param system, but since snapped == snapped on the second
            # pass, the `if snapped != event.new` guard prevents infinite
            # recursion.
            self.startDatePicker.value = snapped

    def _step_hours(self) -> int:
        return STEP_HOURS.get(self.incrementButtons.value, 6)

    def _update_end_date(self, event=None):
        step_hours = self._step_hours()
        n_steps = self.leadTimeSlider.value
        start = self.startDatePicker.value

        if start is not None:
            self.end_date = start + timedelta(hours=step_hours * n_steps)

        total_hours = step_hours * n_steps
        end_str = self.end_date.strftime('%Y-%m-%d %H:%M:%S') if self.end_date else 'N/A'
        self._summary.object = (
            f"**Lead Time:** {n_steps} step{'s' if n_steps != 1 else ''} "
            f"({total_hours} hour{'s' if total_hours != 1 else ''}) &nbsp;&nbsp; "
            f"**End Date:** {end_str}"
        )

    def panel(self):
        return pn.WidgetBox(
            "# Time Settings",
            self.startDatePicker,
            pn.Column(
                self.incrementLabel,
                self.incrementButtonsGroup,
                margin=(0, 0, 0, 10),
                sizing_mode='stretch_width'
            ),
            pn.Column(
                self.leadTimeLabel,
                self.leadTimeSlider,
                margin=(0, 0, 0, 10),
                sizing_mode='stretch_width'
            ),
            self._summary,
        )
