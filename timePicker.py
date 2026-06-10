import panel as pn
import param

from datetime import datetime, timedelta

class TimePicker(param.Parameterized):
    #startDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0))
    #endDate = param.Date(default=datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72))
    #increment = param.Integer(default=1)

    def __init__(self, **params):
        super().__init__(**params)

        self.startDatePicker = pn.widgets.DatetimePicker(
            name="Start Date",
            value = datetime.now().replace(minute=0, second=0, microsecond=0)
            #value=self.startDate
        )
        #self.startDatePicker.link(self, value='startDate')

        self.endDatePicker = pn.widgets.DatetimePicker(
            name="End Date",
            value = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
            #value = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=72)
            #value=self.endDate
        )
        #self.endDatePicker.link(self, value='endDate')

        self.incrementLabel = pn.pane.Markdown("Time Step Increment", margin=(0,0,-5,0))
        self.incrementButtons = pn.widgets.RadioButtonGroup(
            #name="Timestep Increment",
            name="",
            options={'1 hour':'1h', '6 hour':'6h', '12 hour':'12h', '24 hour':'24h'},
            button_type='primary',
            button_style='outline',
            margin=(0,5,5,0)
        )
        self.incrementButtonsGroup = pn.Row(
            #pn.pane.Markdown("Time Step Increment"),
            self.incrementButtons
        )

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
