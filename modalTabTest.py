import panel as pn

def test_modal(event):
    print("     cmdRunner TEMPLATE:", template)
    #template.modal.objects = [
    #template.modal[:] = [
    template.modal.append(
        pn.Column(
            pn.pane.Markdown("## THIS SHOULD SHOW0", height=200),
            height=500,
            #sizing_mode="stretch_width"
        )
    )
    template.open_modal()
#    tmpl = pn.state.template
#    print("     cmdRunner TEMPLATE:", tmpl)
#    #tmpl.modal.objects = [
#    tmpl.modal[:] = [
#        pn.Column(
#            pn.pane.Markdown("## THIS SHOULD SHOW", height=200),
#            height=500,
#            #sizing_mode="stretch_width"
#        )
#    ]
#    tmpl.open_modal()

button = pn.widgets.Button(name="Test Modal", button_type="danger")
button.on_click(test_modal)

w1 = pn.widgets.TextInput(name='Text:')
w2 = pn.widgets.FloatSlider(name='Slider')

modal = pn.Modal(w1, w2, name='Basic FloatPanel', margin=20)
#modal.servable()
toggle_button = modal.create_button("toggle", name="Toggle modal")

#pn.Column('**Example: Basic `Modal`**', toggle_button, modal)

tabs = pn.Tabs(
        ("Tab with modal button", button),
        (pn.pane.Markdown("## THIS SHOULD SHOW1", height=200)),
        #pn.Column(toggle_button, modal)
        #pn.Row(modal, toggle_button)
        pn.Row(toggle_button)
        #("T1", pn.pane.Markdown("## Tab 1", height=200)),
    )

template = pn.template.BootstrapTemplate(title="test modal")
template.modal.append(pn.pane.Markdown("## THIS SHOULD SHOW2", height=200))
print("     created TEMPLATE:", template)

#template.main.append(button)
#col=pn.Column(button,tabs)
#template.main.append(col)
template.main.append(tabs)
#template.main.append(button)
#template.main[:] = [tabs]

template.servable()
