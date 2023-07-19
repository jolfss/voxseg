import omni.ext
import omni.ui as ui

from .voxels import Voxels

from .build_extension import *

# NOTE: Adding elements to a Container must be done by a method *OUTSIDE* the container. 
PAD = 10
TEXT_PAD = ' '*int(PAD/5)

class MyExtension(omni.ext.IExt):
    """The extension object for VoxSeg."""
    voxels : Voxels
    class_labels:dict = {} # TODO: If it turns out grouping labels is important, make this a dictionary that maps key (class label) --> value (grouped labels)

    # Dynamic ui elements
    classes_string_field : ui.StringField
    classes_vstack : ui.VStack

    def build_extension(self) -> ui.Window:
        """Builds the ui elements of the Voxseg Extension."""
        window = ui.Window("Voxseg", width=450, height=700, padding_x=PAD, padding_y=PAD)
        with window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(height=0.0,spacing=PAD):
                    self.build_class_label_editor()
                    self.build_class_labels_vstack()
                    self.build_voxseg_tools()
                    self.build_visualization_tools()
        return window

    def build_class_label_editor(self):
        """Creates the first widget group meant for defining class labels.
        Defines:
            self.class_label_string_field (ui.StringField):
            self.class_label_color_widget (ui.ColorWidget):
        Requires:
            self.class_labels_vstack (ui.VStack): The VStack which will contain all of the class labels."""
        def add_label_fn():
            user_input = self.class_label_string_field.model.as_string.strip()
            color_model  = self.class_label_color_widget.model
            color_list=[(color_model.get_item_value_model(child)).as_float for child in color_model.get_item_children()[:3]]
            color = ui.color(color_list[0],color_list[1],color_list[2])
            if not user_input in self.class_labels.keys():
                self.class_labels.update({user_input:color}) # TODO: Determine type of sublabels.
                self.update_class_labels_vstack()

        with ui.CollapsableFrame("Class Labels"):
            with ui.VStack(height=0):
                ui.Button("add label", clicked_fn=add_label_fn)
                with ui.HStack():
                    self.class_label_string_field = ui.StringField()
                    self.class_label_color_widget = ui.ColorWidget()

    def build_class_labels_vstack(self):
        """Creates the widget which will contain all of the defined class labels and a button to clear them.
        Defines:
            class_labels_vstack (ui.VStack): """
        def clear_labels_fn():
            self.class_labels.clear()
            self.update_class_labels_vstack()

        with ui.CollapsableFrame("View Labels"):
            self.class_labels_vstack = ui.VStack(height=0, spacing=PAD)
        ui.Button("Clear Labels", clicked_fn=clear_labels_fn)

    def update_class_labels_vstack(self):
        """Empties the current vstack of all class labels and rebuilds them from [self.class_labels]
        Requires:
            self.class_labels_vstack (ui.VStack)"""
        print("Update called")
        self.class_labels_vstack.clear()
        with self.class_labels_vstack:
            for class_label in self.class_labels.keys():
                ui.Label(class_label, word_wrap=True, style={"color":self.class_labels[class_label]})

    def build_voxseg_tools(self):
        """Creates the widget which will contain all of the defined class labels.
        Defines:
            class_labels_vstack (ui.VStack): """
        with ui.CollapsableFrame("Voxseg Parameters"):
            with ui.VStack(height=0,spacing=PAD):
                with ui.HStack():
                    ui.Label(F"{TEXT_PAD}World Origin{TEXT_PAD}",width=ui.Fraction(1))
                    ui.MultiFloatDragField(0.0,0.0,0.0,width=ui.Fraction(3))
                with ui.HStack():
                    ui.Label(F"{TEXT_PAD}World Dims{TEXT_PAD}",width=ui.Fraction(1))
                    ui.MultiFloatDragField(1.0,1.0,1.0,min=1.0,step=0.1,width=ui.Fraction(3))
                with ui.HStack():
                    ui.Label(F"{TEXT_PAD}Grid Dims{TEXT_PAD}",width=ui.Fraction(1))
                    ui.MultiIntDragField(2,2,2,min=2,width=ui.Fraction(3))
                    
    def build_visualization_tools(self):
        """TODO: Docs"""
        with ui.CollapsableFrame("Voxel Visualization"):
            with ui.VStack(height=0,spacing=PAD):
                with ui.VStack():
                    ui.Label(F"{TEXT_PAD}Total Occupied Voxels: <UNIMPLEMENTED>")
                    ui.Label(F"{TEXT_PAD}Number of Photos: <UNIMPLEMENTED>")
                ui.Button("visualize occupancy")
                ui.Button("segment over labels")
                ui.Button("clear segments")
                ui.Button("hide/show voxels")


    def on_startup(self, ext_id):

        print("[omni.voxseg] VoxSeg on_startup")
        self.voxels = Voxels((10,10,10),(20,20,20))

        self.window = self.build_extension()            

    def on_shutdown(self):
        print("[omni.voxseg] VoxSeg on_shutdown")
