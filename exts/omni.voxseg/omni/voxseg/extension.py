from typing import Dict, Tuple
import numpy as np
import torch
from torch import Tensor

import omni.ext
import omni.ui as ui
from omni.ui import AbstractValueModel, AbstractItemModel

from .voxels import Voxels

# NOTE: Adding elements to a Container must be done by a method *OUTSIDE* the container. 
PAD = 10
TEXT_PAD = ' '*int(PAD/5)

class MyExtension(omni.ext.IExt):
    """The extension object for VoxSeg."""

    voxels : Voxels
    "The container object for voxel data."

    def build_extension(self) -> ui.Window:
        """Builds the ui elements of the Voxseg Extension."""
        window = ui.Window("Voxseg", width=450, height=700, padding_x=PAD, padding_y=PAD)
        with window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(height=0.0,spacing=PAD):
                    self.build_class_label_editor()
                    self.build_class_vstack()
                    self.build_voxseg_tools()
                    self.build_visualization_tools()
        return window
    
    #-------------------------#
    #   class labels widgets  #
    #-------------------------#
    """The fundamental idea is that each color represents a class, so instead of making structure which groups labels
    under a particular class, simply group all labels by the color they were defined with."""

    color_to_sublabels_dict : dict = {}
    """Groups sublabels by their color, their de facto class. (r,g,b) (float * float * float) -> labels (str list)
    NOTE: The key can be any valid argument to ui.color(), which has many overloads."""

    label_to_color_dict : dict = {}
    "label (str) -> class color (int==ui.ColorShade)"

    def build_class_label_editor(self):
        """Creates the first widget group meant for defining class labels."""

        def get_current_color_and_label() -> Tuple[Tuple[float,float,float],str]:
            """reads in the label and color in the ui and returns them"""
            user_input = self.class_label_string_field.model.as_string.strip()
            color_model  = self.class_color_widget.model
            r,g,b=[(color_model.get_item_value_model(child)).as_float for child in color_model.get_item_children()[:3]]
            return (r,g,b), user_input

        def create_new_class():
            """Changes the color (creates a new class)."""
            (r,g,b), _ = get_current_color_and_label()      

            # make random colors until one is not in the current list of colors
            while (r,g,b) in self.color_to_sublabels_dict.keys():
                r,g,b = np.random.rand(3)

            # removes all colors with no sublabels, NOTE: Not the best way to do this but usually there are few colors
            for color in list(self.color_to_sublabels_dict.keys()):
                if len(self.color_to_sublabels_dict[color]) == 0:
                    self.color_to_sublabels_dict.pop(color) 

            # create new color with no labels
            self.color_to_sublabels_dict.update({(r,g,b):[]})

            # change the color-picker widget to match the generated color
            widget = self.class_color_widget.model
            rmodel,gmodel,bmodel=[(widget.get_item_value_model(child)) for child in widget.get_item_children()[:3]]
            rmodel.set_value(r)
            gmodel.set_value(g)
            bmodel.set_value(b)
            
            self.update_class_vstack()

        def create_new_sublabel(required_for_end_edit):
            """Adds a new label for segmentation to the current color (class)."""
            (r,g,b), label = get_current_color_and_label()

            if len(label) < 1: # one character is already borderline nonsense, but as long as there's something...
                return
            
            if label in self.label_to_color_dict.keys(): 
                print(F"[voxseg] Warning: Label {label} is already reserved.")
                return

            if not (r,g,b) in self.color_to_sublabels_dict.keys(): # handle case where label is added before class
                self.color_to_sublabels_dict.update({(r,g,b):[]})
            
            self.label_to_color_dict.update({label:(r,g,b)}) 

            sublabels : list = self.color_to_sublabels_dict[(r,g,b)] 
            sublabels.append(label)

            self.update_class_vstack()

        with ui.CollapsableFrame("Class Labels"):
            STARTING_COLOR = 0.8,0.2,0.1
            with ui.VStack(height=0):
                with ui.HStack():
                    ui.Button("Change Color (New Class)", clicked_fn=create_new_class)
                    self.class_color_widget = ui.ColorWidget(*STARTING_COLOR)
                    ui.AbstractItemModel.add_end_edit_fn(self.class_color_widget.model, create_new_class)

                self.button_assign_new_sublabel = ui.Button("Assign New Sublabel", clicked_fn=create_new_sublabel)
                self.class_label_string_field = ui.StringField()

                # NOTE: Preferable to button if there is a way to trigger on pressing enter, deselecting is annoying.
                #ui.AbstractValueModel.add_end_edit_fn(self.class_label_string_field.model, create_new_sublabel)
                

    def build_class_vstack(self):
        """Contains all of the registered labels, has a button to clear them."""

        def clear_labels_fn():
            """Empties the current list of labels and colors (classes) and reloads their container."""
            self.color_to_sublabels_dict.clear()
            self.label_to_color_dict.clear()
            self.update_class_vstack()

        with ui.CollapsableFrame("View Labels"):
            with ui.VStack():
                self.class_vstack = ui.VStack(height=0, spacing=PAD)
                ui.Button("Clear Labels", clicked_fn=clear_labels_fn)

    def update_class_vstack(self):
        """Clears and then rebuilds the vstack with all colors (classes) and their labels."""
        self.class_vstack.clear()
        with self.class_vstack:
            for class_color in self.color_to_sublabels_dict.keys():
                for sublabel in self.color_to_sublabels_dict[class_color]:
                    ui.Label(sublabel, style={"font_size": 40.0, "color":ui.color(*class_color)})

    #-------------------------#
    #   voxseg-tools widgets  #
    #-------------------------#
    
    # TODO: Set up editing.
    def build_voxseg_tools(self):
        """Creates the widget which will set voxseg parameters."""
        with ui.CollapsableFrame("Voxseg Parameters"):
            with ui.VStack(height=0,spacing=PAD):
                with ui.HStack():
                    ui.Label(F"{TEXT_PAD}World Origin{TEXT_PAD}",width=ui.Fraction(1))
                    ui.MultiFloatDragField(0.0,0.0,0.0,width=ui.Fraction(3))
                with ui.HStack():
                    ui.Label(F"{TEXT_PAD}World Dims{TEXT_PAD}",width=ui.Fraction(1))
                    ui.MultiFloatDragField(20.0,20.0,20.0,min=1.0,step=0.1,width=ui.Fraction(3))
                with ui.HStack():
                    ui.Label(F"{TEXT_PAD}Grid Dims{TEXT_PAD}",width=ui.Fraction(1))
                    ui.MultiIntDragField(10,10,10,min=2,width=ui.Fraction(3))

    #---------------------------------#
    #   voxseg-visualization widgets  #
    #---------------------------------#

    def __debug_create_all_voxels(self):
        # self.voxels.create_voxel_prims(
        #     torch.arange(min(self.voxels.grid_dims)).expand(3,min(self.voxels.grid_dims)).T.clone())
        gx,gy,gz = self.voxels.grid_dims
        GX, GY, GZ = torch.arange(gx),torch.arange(gy),torch.arange(gz)
        GXE, GYE, GZE = GX.expand(gz,gy,-1), GY.expand(gx,gz,-1), GZ.expand(gx,gy,-1)
        GXEP, GYEP = GXE.permute(2,1,0), GYE.permute(0,2,1)
        all_voxels = torch.stack((GXEP,GYEP,GZE),dim=3).view(-1,3)
        self.voxels.create_voxel_prims(all_voxels)

    def visualize_occupancy_fn(self):
        


    def build_visualization_tools(self):
        """TODO: Docs"""
        with ui.CollapsableFrame("Voxel Visualization"):
            with ui.VStack(height=0,spacing=PAD):
                with ui.VStack():
                    ui.Label(F"{TEXT_PAD}Total Occupied Voxels: <UNIMPLEMENTED>")
                    ui.Label(F"{TEXT_PAD}Number of Photos: <UNIMPLEMENTED>")
                ui.Button("visualize occupancy", clicked_fn=self.visualize_occupancy_fn) #TODO:
                ui.Button("segment over labels")
                ui.Button("clear segments")
                ui.Button("hide/show voxels")

    def on_startup(self, ext_id):

        print("[omni.voxseg] VoxSeg on_startup")
        self.voxels = Voxels((10,10,10),(20,20,20))

        self.window = self.build_extension()            

    def on_shutdown(self):
        print("[omni.voxseg] VoxSeg on_shutdown")
