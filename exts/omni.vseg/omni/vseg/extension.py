import omni.ext
import omni.ui as ui

# Class Label Management System
EMPTY_CLASS_LABELS_PLACEHOLDER = "--empty"

def clear_labels_fn(class_labels : set, display_label : ui.Label):
    """Empties [class_labels] and sets [display_label] to [EMPTY_CLASS_LABELS_PLACEHOLDER]."""
    class_labels.clear()
    display_label.text = EMPTY_CLASS_LABELS_PLACEHOLDER

def add_new_class_label_fn(class_labels : set, input_field : ui.StringField, display_label : ui.Label):
    """Adds or removes a class label from [class-labels], taking the label from [input_field]
    and writing all current labels to [display_label] or the [EMPTY_CLASS_LABELS_PLACEHOLDER] if no labels"""
    user_input = input_field.model.as_string.strip()

    if len(user_input) < 1: # Labels are only useful if there is at least one valid letter.
        return 
    
    if user_input[0] == '-': # Delete class label
        user_input = user_input[1:]
        if user_input in class_labels:
            class_labels.remove(user_input)
    elif not user_input in class_labels: # Add class label
        class_labels.add(user_input)
    
    if len(class_labels) == 0: 
        display_label.text = EMPTY_CLASS_LABELS_PLACEHOLDER
        return

    label_text = ""
    for class_label in class_labels:
        label_text = F"{label_text}\t\t\t{class_label}"

    display_label.text = label_text

# Segmentation System
def compute_segment_fn():
    pass

def cleanup_fn():
    pass

def toggle_prims_fn():
    pass


class MyExtension(omni.ext.IExt):
    """The extension object for VSeg."""

    class_labels            : set = set()       # contains all of the class labels registered
    class_label_input_field : ui.StringField    # user input field for adding/removing class labels
    class_labels_label      : ui.Label          # shows the labels the user has currently registered


    def on_startup(self, ext_id):
        print("[omni.vseg] VSeg on_startup")

        self._window = ui.Window("VSeg", width=450, height=700)
        with self._window.frame:
            with ui.VStack():
                with ui.VStack():
                    with ui.VStack(): 
                        self.class_label_input_field = ui.StringField()
                        ui.Button("Add new class label (or remove with -LABEL)", clicked_fn=lambda : add_new_class_label_fn(self.class_labels, self.class_label_input_field, self.class_labels_label))
                    self.class_labels_label = ui.Label(EMPTY_CLASS_LABELS_PLACEHOLDER, alignment=ui.Alignment.CENTER, word_wrap=True)
                    ui.Button("Clear Labels", clicked_fn=lambda : clear_labels_fn(self.class_labels, self.class_labels_label))
                ui.Button("Compute Segments", clicked_fn=compute_segment_fn)
                ui.Button("Clear Segments", clicked_fn=cleanup_fn)
                ui.Button("Show/Hide Voxels", clicked_fn=toggle_prims_fn)
                
                # No other ui elements can occur backdented from here.

    def on_shutdown(self):
        print("[omni.vseg] VSeg on_shutdown")
