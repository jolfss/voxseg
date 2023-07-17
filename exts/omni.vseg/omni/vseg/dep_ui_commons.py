from typing import Set
import omni.ui as ui
from omni.ui import AbstractValueModel as avm
from omni.ui import AbstractItemModel as aim

DEFAULT_LABEL_HEIGHT = 0
DEFAULT_LABEL_WIDTH = 0

#AbstractValueModel Triggers
VALUE_CHANGED = avm.add_value_changed_fn
END_EDIT = avm.add_end_edit_fn

#AbstractItemModel Triggers
ITEM_CHANGED = aim.add_item_changed_fn


def write_field(field: ui.AbstractValueModel, value):
    """Writes the value into the field given.
    NOTE: Must have the .model inputted or at least have a .set_value() method."""
    field.set_value(value)


def read_int(field: ui.AbstractValueModel):
    """Reads the int component of the field.
    NOTE: A common error is to pass in the UI element instead of the .model"""
    return(field.as_int)


def read_float(field: ui.AbstractValueModel):
    """Reads the float component of the field.
    NOTE: A common error is to pass in the UI element instead of the .model"""
    return(field.as_float)


def read_bool(field: ui.CheckBox):
    """Reads the bool component of the field.
    NOTE: A common error is to pass in the UI element instead of the .model"""
    return(field.as_bool)


def read_string(field: ui.StringField):
    """Reads the string component of the field.
    NOTE: A common error is to pass in the UI element instead of the .model"""
    return(field.as_string)


def separate(pre_height=0, post_height=0):
    """Creates a separator with a before and after spacing."""
    return (ui.Label("", height=pre_height), ui.Separator(height=post_height))


def labeled_label(name, height=DEFAULT_LABEL_HEIGHT, width=DEFAULT_LABEL_WIDTH, *register_fns):
    """Creates two labels, the second of which is returned and the first is written with its name."""
    ui.Label(name, height=0, width=width)
    new_element = ui.Label("", height=height, width=width)
    for trigger, response in register_fns:
        trigger(new_element, response)
    return new_element

# Triggers for these can be found at the top of this file, the responses are lambdas which are most always specified
# by the user wherever the UI element is created.

def int_slider(name, min=0, max=10, *register_fns):
    """Creates an int slider with a min, max, and event bindings."""
    ui.Label(name)
    new_element = ui.IntSlider(min=min, max=max)
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def string_slider(name, options, *register_fns):
    """Creates a string slider (labeled int slider, 1 value in range per string)."""
    register_fns = list(register_fns)
    selector = ui.Label(name)
    with ui.HStack():
        for i in range(0, len(options)):
            ui.Label(options[i], alignment=ui.Alignment.CENTER, )
    new_element = ui.IntSlider(min=0, max=len(options)-1)
    new_element.set_style({"color" : 0x00FFFFFF})
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def int_field(name, min=0, max=10, *register_fns):
    """Creates an int field."""
    ui.Label(name)
    new_element = ui.IntField(min=min, max=max)
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def float_field(name, min=0, max=10, *register_fns):
    """Creates a float field."""
    ui.Label(name)
    new_element = ui.FloatField(min=min, max=max)
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def string_field(name, *register_fns):
    """Creates a string field."""
    ui.Label(name)
    new_element = ui.StringField()
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def button(name, clicked_fn, height=50, *register_fns):
    """Creates a button with a clicked function and other event trigger responses."""
    new_element = ui.Button(text=name, height=height, clicked_fn=lambda : clicked_fn())
    for trigger, response in register_fns:
        trigger(new_element, response)
    return new_element


def check_box(name, *register_fns):
    """Creates a check_box with trigger responses."""
    ui.Label(name)
    new_element = ui.CheckBox()
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def combo_box(name, options, *register_fns):
    """Creates a combo_box with trigger responses."""
    ui.Label(name)
    new_element = ui.ComboBox(0, *(options))
    for trigger, response in register_fns:
        trigger(new_element.model, response)
    return new_element


def menu(name, menu_style={}, height=0, *register_fns):
    """Creates a menu with trigger responses."""
    new_element = ui.Menu(text=name, style=menu_style, height=height)
    for trigger, response in register_fns:
        trigger(new_element.delegate, response)
    return new_element