# general python
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor

# omniverse
import pxr
from pxr import UsdGeom, Gf
import omni.ext
import omni.ui as ui
from omni.ui import \
    Window, CollapsableFrame, ScrollingFrame, VStack, HStack \
    , Label, StringField, ColorWidget, Button \
    , AbstractValueModel, AbstractItemModel, AbstractItem, MultiFloatDragField, MultiIntDragField\
    , Fraction

# library
from .voxels import Voxels

# NOTE: Adding elements to a Container must be done by a method *OUTSIDE* the container. 

PAD = 10
TXTPAD = ' '*int(PAD/5)

DEFAULT_VOXEL_CENTER = (0., 0., 0.)
DEFAULT_WORLD_DIMS =   (20.,20.,8.)
DEFAULT_GRID_DIMS  =   (20, 20, 8)


class MyExtension(omni.ext.IExt):
    """The extension object for voxvis."""

    voxels : Voxels = Voxels(DEFAULT_GRID_DIMS, DEFAULT_WORLD_DIMS)
    preview_voxels : Voxels = Voxels(DEFAULT_GRID_DIMS, DEFAULT_WORLD_DIMS)

    "The container object for voxel data."
    "TODO"
   
    def on_startup(self, ext_id):
        """TODO: Describe the order of initialization sensitivities. (what depends on what)"""
        print("[omni.voxvis] voxvis on_startup")
        self.voxels = Voxels(DEFAULT_GRID_DIMS, DEFAULT_WORLD_DIMS)

        self.window = self.build_extension()                   

    def on_shutdown(self):
        """TODO: """
        print("[omni.voxvis] voxvis on_shutdown")

    def build_extension(self) -> Window:
        """Builds the ui elements of the voxvis Extension."""
        window = Window("voxvis", width=450, height=700, padding_x=PAD, padding_y=PAD)
        with window.frame:
            with ScrollingFrame():
                with VStack(height=0.0,spacing=PAD):
                    self.build_domain_editor()
                    self.build_class_label_editor()
                    self.build_class_vstack()
                    self.build_visualization_tools()
        return window
    
    #----------------------------------------------------------------#
    #   this block deals with specifying the domains of the voxels   #
    #----------------------------------------------------------------#
    """
    These widgets allow the user to define where they want their voxel grid to be before anything is done with voxels.
    However, this gets disabled after the self.voxels parameter is set because the location and size of the voxels
    cannot be altered afterwards.
    """

    def build_domain_editor(self):
        """Creates the widget which will set voxvis parameters."""
        with CollapsableFrame("voxvis Parameters"):
            with VStack(height=0,spacing=PAD):
                with HStack():
                    Label(F"{TXTPAD}Voxel Center{TXTPAD}",width=Fraction(1))
                    self.multi_float_voxel_center = MultiFloatDragField(*DEFAULT_VOXEL_CENTER,width=Fraction(3))         
                with HStack():
                    Label(F"{TXTPAD}Grid Dims{TXTPAD}",width=Fraction(1))
                    self.multi_int_grid_dims = MultiIntDragField(*DEFAULT_GRID_DIMS,min=2,width=Fraction(3))           
                with HStack():
                    Label(F"{TXTPAD}World Dims{TXTPAD}",width=Fraction(1))
                    self.multi_float_world_dims = MultiFloatDragField(*DEFAULT_WORLD_DIMS,min=1,step=0.1,width=Fraction(3))


        self.apply_preview_callbacks()

    def get_domain_value_models(self):
        """Returns (as value models):
            (cx,cy,cz), (gx,gy,gz), (wx,wy,wz),"""
        model = self.multi_float_voxel_center.model
        voxel_center=[(model.get_item_value_model(child)) for child in model.get_item_children()[:3]]

        model = self.multi_int_grid_dims.model
        grid_dims=[(model.get_item_value_model(child)) for child in model.get_item_children()[:3]]

        model = self.multi_float_world_dims.model
        world_dims=[(model.get_item_value_model(child)) for child in model.get_item_children()[:3]]

        return tuple(voxel_center), tuple(grid_dims), tuple(world_dims)

    def get_domain_values(self):
        """Returns (as values):
            (cx,cy,cz), (gx,gy,gz), (wx,wy,wz)"""
        (cx,cy,cz), (gx,gy,gz), (wx,wy,wz)  = self.get_domain_value_models()

        cx, cy, cz = cx.as_float, cy.as_float, cz.as_float
        gx, gy, gz = gx.as_int,   gy.as_int,   gz.as_int
        wx, wy, wz = wx.as_float, wy.as_float, wz.as_float


        return (cx,cy,cz), (gx,gy,gz), (wx,wy,wz)

    def apply_preview_callbacks(self):
        """Previews the voxel space."""
        enable_preview_fn = lambda _: self.preview_voxels.toggle(True)

        def while_previewing_fn(dummy=None):
            _,(gx,gy,gz),world_dims= self.get_domain_values()  

            self.preview_voxels.redomain((gx,gy,gz),world_dims)

            self.preview_voxels.register_new_voxel_color((0,0,0))
            self.preview_voxels.register_new_voxel_color((1,1,1))
            
            shell_indices = self.preview_voxels.indices_shell()
            checkerboard_classes = (shell_indices[:,0] + shell_indices[:,1] + shell_indices[:,2]) % 2
            #shell_indices = self.preview_voxels.indices().view(-1,3)
            #checkerboard_classes = (shell_indices[:,0] + shell_indices[:,1] + shell_indices[:,2]) % 2
            self.preview_voxels.create_voxels(shell_indices, checkerboard_classes)

        disable_preview_fn = lambda _ : self.preview_voxels.toggle(False)

        (cx,cy,cz), (gx,gy,gz), (wx,wy,wz) = self.get_domain_value_models()
        for model in [cx,cy,cz,gx,gy,gz,wx,wy,wz]:
            model.add_begin_edit_fn(enable_preview_fn)
            model.add_value_changed_fn(while_previewing_fn)
            model.add_end_edit_fn(disable_preview_fn)

    def disable_domain_editing(self):
        """TODO: Docs"""
        self.multi_float_voxel_center.enabled = False
        self.multi_float_world_dims.enabled = False
        self.multi_int_grid_dims.enabled = False
        
    #-------------------------------------------#
    #   this block deals with defining labels   #
    #-------------------------------------------#
    """
    The fundamental idea is that each color represents a class, so instead of making structure which groups labels
    under a particular class, simply group all labels by the color they were defined with.

    There are some tags which the user can use, these must be specified with the first label in the class.
    -i --invisible, makes the voxel invisible by default (sets the prototype voxel prim to invisible)
    """

    default_class_labels : str = ["not labeled --invisible"]
    "Represents the empty class."

    default_class_color : Tuple[float,float,float] = (1/255,1/255,1/255) # People are way more likely to choose #000000
    "This color is reserved for the non-label, attempts to use it will fail."

    dict_color_to_label : dict = {default_class_color : default_class_labels}
    "Groups sublabels by their color, their de facto class. r,g,b (float * float * float) -> labels (str list)"

    dict_label_to_color : dict = {}
    "label (str) -> r,g,b (float * float * float)"

    def register_default_class(self):
        """Registers the default class to the appropriate dictionaries."""

        # Update the color to point to the labels, if any, which denote the default class.
        self.dict_color_to_label = {self.default_class_color : self.default_class_labels}

        # Connect each label in the default labels to the default class color.
        for default_label in self.default_class_labels:
            self.dict_label_to_color.update({default_label : self.default_class_color})        

    def get_current_color_and_label(self) -> Tuple[Tuple[float,float,float],str]:
        """reads in the label and color in the ui and returns them"""
        user_input = self.string_field_class_label.model.as_string.strip()
        color_model  = self.color_widget_class_color.model
        r,g,b=[(color_model.get_item_value_model(child)).as_float for child in color_model.get_item_children()[:3]]
        return (r,g,b), user_input

    def load_classes_from_dictionary(self, dict_color_to_labels : Dict[Tuple[float,float,float],List[str]]):
        for color in dict_color_to_labels.keys():
            self.create_new_class(color_override=color)
            for label in dict_color_to_labels[color]:
                self.create_new_sublabel(label_override=label)

    def create_new_class(self, color_override : Optional[Tuple[float,float,float]]=None):
        """Changes the color (creates a new class).
        Args:
            color_override ((float * float * float) option): Color to use for the class aside from the ui widget
            NOTE: Useful for loading in data/ code access point.
        TODO: Requires:"""

        # read in from the ui the current color OR get from arguments (usually for user loading in data)
        (r,g,b), _ = self.get_current_color_and_label()

        if color_override:
            (r,g,b) = color_override

        # make random colors until one is not in the current list of colors
        while (r,g,b) in self.dict_color_to_label.keys():
            print("[voxvis] Warning: A class tried to use a color which was already reserved, assigning a new color.")
            r,g,b = np.random.rand(3)

        # removes all colors with no sublabels, NOTE: Not the best way to do this but usually there are few colors
        for color in list(self.dict_color_to_label.keys()):
            if len(self.dict_color_to_label[color]) == 0:
                self.dict_color_to_label.pop(color) 

        # create new color with no labels
        self.dict_color_to_label.update({(r,g,b):[]})

        # change the color-picker widget to match the generated color
        widget = self.color_widget_class_color.model
        rmodel,gmodel,bmodel=[(widget.get_item_value_model(child)) for child in widget.get_item_children()[:3]]
        rmodel.set_value(r)
        gmodel.set_value(g)
        bmodel.set_value(b)

    def create_new_sublabel(self, label_override : Optional[str]=None):
        """Adds a new label for segmentation to the *current color* (class).
        Args:
            label_override (str option): Color to use for the class aside from the ui widget
            NOTE: Useful for loading in data/ code access point."""

        (r,g,b), label = self.get_current_color_and_label()

        label = label_override if label_override is not None else label

        if len(label) < 1: # one character is already borderline nonsense, but as long as there's something...
            return
        
        if (r,g,b) == self.default_class_color:
            print(F"[voxvis] Warning: The color {self.default_class_color} is permanently reserved. Randomizing.")
            self.create_new_class()
            return
        
        if label in self.dict_label_to_color.keys(): 
            print(F"[voxvis] Warning: Label {label} is already reserved.")
            return

        if not (r,g,b) in self.dict_color_to_label.keys(): # handle case where label is added before class
            self.dict_color_to_label.update({(r,g,b):[]})
        
        self.dict_label_to_color.update({label:(r,g,b)}) 

        sublabels : list = self.dict_color_to_label[(r,g,b)] 
        sublabels.append(label)

        self.update_class_vstack()

    def build_class_label_editor(self):
        """Creates the first widget group meant for defining class labels."""
        with CollapsableFrame("Class Label Editor"):
            STARTING_COLOR = 0.8,0.2,0.1
            with VStack(height=0):
                with HStack():
                    Button("Change Color (New Class)", clicked_fn=self.create_new_class)
                    self.color_widget_class_color = ColorWidget(*STARTING_COLOR)
                    AbstractItemModel.add_end_edit_fn(self.color_widget_class_color.model, lambda _, __ : self.create_new_class())

                self.button_assign_new_sublabel = Button("Assign New Sublabel", clicked_fn=self.create_new_sublabel)
                self.string_field_class_label = StringField()

                # NOTE: Preferable to button if there is a way to trigger on pressing enter, deselecting is annoying.
                # AbstractValueModel.add_end_edit_fn(self.class_label_string_field.model, create_new_sublabel)
                
    #----------------------------------------------------------------#
    #   this block lets the user see what labels have been defined   #
    #----------------------------------------------------------------#
    """
    Shows the user all of the current classes (and their labels) that they have defined so far, in order.
    The first label in each class is slightly bigger to denote that it is the principle label.
    """

    def build_class_vstack(self):
        """Contains all of the registered labels, has a button to clear them."""

        def clear_labels_fn():
            """Empties the current list of labels and colors (classes) and reloads their container."""
            self.dict_color_to_label.clear()
            self.dict_label_to_color.clear()

            # Ensure default label is added first.
            self.register_default_class()

            self.update_class_vstack()
            

        with CollapsableFrame("View Labels"):
            with VStack():
                self.class_vstack = VStack(height=0, spacing=PAD)
                Button("Clear Labels", clicked_fn=clear_labels_fn)

    def update_class_vstack(self):
        """Clears and then rebuilds the vstack with all colors (classes) and their labels."""
        self.class_vstack.clear()
        with self.class_vstack:
            for class_color in self.dict_color_to_label.keys():
                first = True
                for class_label in self.dict_color_to_label[class_color]:
                    Label(class_label, style={"font_size": 28.0, "color":ui.color(*class_color)} if first else {"font_size": 20.0, "color":ui.color(*class_color)})
                    first = False

    def get_class_colors(self) -> List[Tuple[float,float,float]]:
        """Returns all of the colors which correspond to existing classes."""
        class_colors = list(self.dict_color_to_label.keys())
        return class_colors
    
    #-----------------------------------------------------------------#
    #   this block has buttons that make voxels appear in the stage   #
    #-----------------------------------------------------------------#
    """
    TODO
    """
    def build_visualization_tools(self):
        """TODO: Docs"""
        with CollapsableFrame("Voxel Visualization"):
            with VStack(height=0,spacing=PAD):
                with VStack():
                    Label(F"{TXTPAD}Total Occupied Voxels: <UNIMPLEMENTED>")
                    Label(F"{TXTPAD}Number of Photos: <UNIMPLEMENTED>")
                Button("--DEBUG load classes from dictionary", clicked_fn=self.__DEMO__load_custom_classes)
                Button("--DEBUG randomize over current labels", clicked_fn=self.__DEMO__create_randomly_labeled_voxels)
                Button("visualize occupancy", clicked_fn=lambda : breakpoint())
                Button("segment over labels")
                Button("clear segments")
                Button("hide/show voxels", clicked_fn=lambda : self.voxels.toggle())

    def show_voxels(self, voxel_indices : Tensor, voxel_classes : Tensor):
        """Populates the world with all of the voxels specified, removing any that were there before.
        Args:
            voxel_indices (N,3): Each row is an ijk of a voxel to show. Duplicates are not forbidden but cause overlap.
            voxel_classes (N,):  Each element is a class index [0...number of registered classes).
        """
        self.disable_domain_editing()
        self.voxels.create_voxels(voxel_indices, voxel_classes)

    def __DEMO__create_randomly_labeled_voxels(self):
        # Create set of all indices in the voxel grid then reshape to (N,3)
        all_voxel_indices = self.voxels.indices()

        # Register a new voxel color per class.
        class_colors = self.dict_color_to_label.keys()
        for color in class_colors:
            i_tag = "--invisible" in  self.dict_color_to_label[color][0] or " -i" in self.dict_color_to_label[color][0]
            class_index = self.voxels.register_new_voxel_color(color, invisible=i_tag)
            # NOTE: Typically you keep track of this class <-> index, we don't here because we randomly assign classes.

        # Indices of classes are integers \in [0,num_classes). 
        random_classes = (torch.rand(self.voxels.capacity()) * len(class_colors)).floor().int()
        
        # The color of each class was set during registration (when the protovoxel was created).
        # See the above NOTE for figuring out what index to pass for each class.
        self.show_voxels(all_voxel_indices, random_classes)

    def __DEMO__load_custom_classes(self):
        custom_classes = {(0.8,0.5,0.2):["orange_color"],(0.2,0.9,0.2):["green_color","verde","multiple greens baby"],(0.9,0.1,0.1):["blank red --invisible"],(0.8,0.1,0.9):["purple_color"]}
        self.load_classes_from_dictionary(custom_classes)         

    def visualize_occupancy_fn(self):
        pass
