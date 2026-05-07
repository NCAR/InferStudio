import panel as pn
import param

from datetime import datetime, timedelta

class TimePicker(param.Parameterized):
    startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))
    increment = param.Integer(default=1)

    def __init__(self, **params):
        super().__init__(**params)

        self.startDatePicker = pn.widgets.DatetimePicker(
            name="Start Date",
            value=self.startDate
        )
        self.startDatePicker.link(self, value='startDate')

        self.endDatePicker = pn.widgets.DatetimePicker(
            name="End Date",
            value=self.endDate
        )
        self.endDatePicker.link(self, value='endDate')

        self.incrementLabel = pn.pane.Markdown("Time Step Increment", margin=(0,0,-5,0))
        self.incrementButtons = pn.widgets.RadioButtonGroup(
            name="Timestep Increment",
            options={'1 hour':1, '6 hour':6, '12 hour':12, '24 hour':24},
            button_type='default',
            margin=(0,5,5,0)
        )

    def panel(self):
        #return pn.Column(
        return pn.WidgetBox(
            "# Time Settings",
            pn.Row(self.startDatePicker, self.endDatePicker),
            pn.Column(
                self.incrementLabel, 
                self.incrementButtons, 
                margin=(0,0,0,10),
                sizing_mode='stretch_width'
            )
        )
