import panel as pn
import param
from datetime import datetime, timedelta

VALID_CYCLES = {0, 6, 12, 18}

# Check available dates for gfs_init.py with
# gsutil ls gs://global-forecast-system/gdas.20260610/

class TimePicker(param.Parameterized):
    def __init__(self, **params):
        super().__init__(**params)

        self.startDatePicker = pn.widgets.DatetimePicker(
            name="Start Date",
            value = self._snap_to_cycle(datetime.now() - timedelta(hours=24))
            #value = datetime.now().replace(minute=0, second=0, microsecond=0)
        )

        self.endDatePicker = pn.widgets.DatetimePicker(
            name="End Date",
            value = self._snap_to_cycle(datetime.now()) - timedelta(hours=18)
            #value = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
        )
        self.startDatePicker.param.watch(self._snap_start, 'value')
        self.endDatePicker.param.watch(self._snap_end, 'value')

        self.incrementLabel = pn.pane.Markdown("Time Step Increment", margin=(0,0,-5,0))
        self.incrementButtons = pn.widgets.RadioButtonGroup(
            name="",
            options={'1 hour':'1h', '6 hour':'6h', '12 hour':'12h', '24 hour':'24h'},
            value='6h',
            button_type='primary',
            button_style='outline',
            margin=(0,5,5,0)
        )
        self.incrementButtonsGroup = pn.Row(
            self.incrementButtons
        )

    #@staticmethod
    #def _snap_to_cycle(dt: datetime) -> datetime:
    #    """Round datetime to nearest GDAS cycle hour (00/06/12/18Z)."""
    #    h = dt.hour
    #    snapped = min(VALID_CYCLES, key=lambda c: min(abs(h - c), abs(h - c + 24), abs(h - c - 24)))
    #    return dt.replace(hour=snapped, minute=0, second=0, microsecond=0)
    @staticmethod
    def _snap_to_cycle(dt: datetime) -> datetime:
        snapped_hour = (dt.hour // 6) * 6
        return dt.replace(hour=snapped_hour, minute=0, second=0, microsecond=0)

    def _snap_start(self, event):
        snapped = self._snap_to_cycle(event.new)
        if snapped != event.new:
            self.startDatePicker.value = snapped

    def _snap_end(self, event):
        snapped = self._snap_to_cycle(event.new)
        if snapped != event.new:
            self.endDatePicker.value = snapped

    def panel(self):
        #return pn.Column(
        return pn.WidgetBox(
            "# Time Settings",
            pn.Row(self.startDatePicker, self.endDatePicker),
            pn.Column(
                self.incrementLabel, 
                self.incrementButtonsGroup, 
                margin=(0,0,0,10),
                sizing_mode='stretch_width'
            )
        )
