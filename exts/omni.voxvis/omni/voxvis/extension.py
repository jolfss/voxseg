# general python
from typing import Callable, Dict, List, Optional, Tuple
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
    , AbstractItemModel, AbstractValueModel \
    , MultiFloatDragField, MultiIntDragField\
    , Fraction

# library
from .voxels import Voxels
from .client import VoxSegClient

# NOTE: Adding elements to a Container must be done by a method *OUTSIDE* the container. 

PAD = 10
TXTPAD = ' '*int(PAD/5)

DEFAULT_VOXEL_CENTER = (0., 0., 0. ) # FOR NOW DEFAULTS 
DEFAULT_GRID_DIMS  =   (40, 40, 10  )
DEFAULT_WORLD_DIMS =   (20.,20.,5.)


class MyExtension(omni.ext.IExt):
    """The extension object for voxvis."""
    def __DEMO__randomize_over_classes(self):
        # Create set of all indices in the voxel grid then reshape to (N,3)
        all_voxel_indices = self.voxels.indices()
        # Indices of classes are integers \in [0,num_classes). 
        random_classes = (torch.rand(self.voxels.capacity()) * len(self.voxels.classes)).floor().int()
        
        # The color of each class was set during registration (when the protovoxel was created).
        # See the above NOTE for figuring out what index to pass for each class.
        self.voxels.create_voxels(all_voxel_indices, random_classes)
        self.voxels.toggle(True)

    def __DEMO__load_dictionary(self):
        def load_dictionary(dictionary):
            for color in dictionary.keys():
                self.voxels.create_class(dictionary[color][0], color)

        custom_classes = {(0.6, 0.4, 0.2): ['ground'],
                          (0.5, 0.5, 0.5): ['rock'],
                          (0.7, 0.2, 0.2): ['brick'],
                          (0.8, 0.0, 0.0): ['fire_hydrant'],
                          (0.035, 0.03, 0.2): ['heavy_machinery']}
        load_dictionary(custom_classes)
        self.update_class_vstack()

    def visualize_occupancy_fn(self):
        pass
    
    #--------------------------------------------------------------------------------------#
    #   this block deals with creating all of the ui elements (first point of execution)   #
    #--------------------------------------------------------------------------------------#
    def on_startup(self, ext_id):
        """TODO: Describe the order of initialization sensitivities. (what depends on what)"""
        print("[omni.voxvis] voxvis on_startup")
        self.voxels = Voxels(DEFAULT_WORLD_DIMS, DEFAULT_GRID_DIMS, directory="/World/voxvis/voxels")
        
        self.preview_voxels = Voxels(DEFAULT_WORLD_DIMS, DEFAULT_GRID_DIMS, directory="/World/voxvis/preview")

        self.window = self.build_extension()   
        self.client = VoxSegClient()               

        self.client.publish_world_info(self.voxels.world_dims,self.voxels.grid_dims) 

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
                    Label(F"{TXTPAD}World Dims{TXTPAD}",width=Fraction(1))
                    self.multi_float_world_dims = MultiFloatDragField(*DEFAULT_WORLD_DIMS,min=1,step=0.1,width=Fraction(3))
                with HStack():
                    Label(F"{TXTPAD}Grid Dims{TXTPAD}",width=Fraction(1))
                    self.multi_int_grid_dims = MultiIntDragField(*DEFAULT_GRID_DIMS,min=2,width=Fraction(3))   
                with HStack():
                    Label(F"{TXTPAD}Voxel Center{TXTPAD}",width=Fraction(1))
                    self.multi_float_voxel_center = MultiFloatDragField(*DEFAULT_VOXEL_CENTER,width=Fraction(3))         
        self.apply_preview_callbacks()

    def get_domain_value_models(self):
        """Returns (as value models):
            (wx,wy,wz), (gx,gy,gz), (cx,cy,cz)"""
        model = self.multi_float_world_dims.model
        world_dims=[(model.get_item_value_model(child)) for child in model.get_item_children()[:3]]

        model = self.multi_int_grid_dims.model
        grid_dims=[(model.get_item_value_model(child)) for child in model.get_item_children()[:3]]

        model = self.multi_float_voxel_center.model
        voxel_center=[(model.get_item_value_model(child)) for child in model.get_item_children()[:3]]

        return tuple(world_dims), tuple(grid_dims), tuple(voxel_center)

    def get_domain_values(self):
        """Returns (as values):
            (wx,wy,wz), (gx,gy,gz), (cx,cy,cz)"""
        (wx,wy,wz), (gx,gy,gz), (cx,cy,cz) = self.get_domain_value_models()

        wx, wy, wz = wx.as_float, wy.as_float, wz.as_float
        gx, gy, gz = gx.as_int,   gy.as_int,   gz.as_int
        cx, cy, cz = cx.as_float, cy.as_float, cz.as_float

        return (wx,wy,wz), (gx,gy,gz), (cx,cy,cz)

    def apply_preview_callbacks(self):
        """Previews the voxel space."""

        def on_begin_preview(dummy=None):
            self.voxels.toggle(False)
            self.preview_voxels.toggle(True)

        def while_previewing_fn(dummy=None):
            self.preview_voxels.clear_classes()
            self.preview_voxels.redomain(*self.get_domain_values())

            self.preview_voxels.create_class("even",(0,0,0))
            self.preview_voxels.create_class("odd", (1,1,1))
            
            shell_indices = self.preview_voxels.shell()
            checkerboard_classes = (shell_indices[:,0] + shell_indices[:,1] + shell_indices[:,2]) % 2
            self.preview_voxels.create_voxels(shell_indices, checkerboard_classes)

        def on_end_preview(dummy=None):
            self.preview_voxels.toggle(False)
            self.voxels.redomain(*self.get_domain_values()) 
            self.client.publish_world_info(self.voxels.world_dims,self.voxels.grid_dims)
            #self.voxels.toggle(True) NOTE: Aesthetic to have it True, but no longer accurate so just disable.

        (wx,wy,wz), (gx,gy,gz), (cx,cy,cz) = self.get_domain_value_models()
        for model in [wx,wy,wz,gx,gy,gz,cx,cy,cz]:
            model.add_begin_edit_fn(on_begin_preview)
            model.add_value_changed_fn(while_previewing_fn)
            model.add_end_edit_fn(on_end_preview)

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
    """      

    def get_current_label(self) -> Tuple[Tuple[float,float,float],str]:
        """reads in the label and color in the ui and returns them"""
        user_input = self.string_field_class_label.model.as_string.strip()
        return user_input
    
    def get_random_color_not_in_use(self, f: Callable[[float],float]=lambda x:x):
        r,g,b = np.random.rand(), np.random.rand(), np.random.rand()
        r,g,b = f(r), f(g), f(b)
        while (r,g,b) in self.voxels.colors:
            r,g,b = f(np.random.rand()), f(np.random.rand()), f(np.random.rand())
        return r,g,b

    def set_widget_color(self, widget : ColorWidget, color:Tuple[float,float,float]):
        rmodel,gmodel,bmodel=[(widget.model.get_item_value_model(child)) for child in widget.model.get_item_children()[:3]]
        r,g,b = color
        rmodel.set_value(r)
        gmodel.set_value(g)
        bmodel.set_value(b)

    def clear_string_field_input(self):
        self.string_field_class_label.model.set_value("")

    def create_class(self, color_override : Optional[Tuple[float,float,float]]=None):
        label, (r,g,b) = self.get_current_label(), self.get_random_color_not_in_use()
        self.voxels.create_class(label, (r,g,b) if color_override is None else color_override)
        self.update_class_vstack()    
        self.clear_string_field_input()

    def add_label(self, label_override : Optional[str]=None):
        label = self.get_current_label()
        self.voxels.add_label(self.voxels.classes[-1], label if label_override is None else label_override)
        self.update_class_vstack()
        self.clear_string_field_input()

    def build_class_label_editor(self):
        """Creates the first widget group meant for defining class labels."""
        with CollapsableFrame("Class Label Editor"):
            with VStack(height=0):
                with ui.HStack():
                    self.button_create_class = Button("Create Class", clicked_fn=self.create_class)
                    self.button_add_label = Button("Assign New Sublabel", clicked_fn=self.add_label)
                self.string_field_class_label = StringField()

                
    #----------------------------------------------------------------#
    #   this block lets the user see and modify classes and labels   #
    #----------------------------------------------------------------#
    """
    Shows the user all of the current classes (and their labels) that they have defined so far, in order.
    The first label in each class is slightly bigger to denote that it is the principle label.
    """

    class_to_rgb_widgets : Dict[str,ColorWidget]= {}

    def build_class_vstack(self):
        """Contains all of the registered labels, has a button to clear them."""

        def clear_labels_fn():
            """Empties the current list of labels and colors (classes) and reloads their container."""
            self.voxels.clear_classes()
            self.update_class_vstack()

        with CollapsableFrame("View Labels"):
            with VStack():
                self.class_vstack = VStack(height=0, spacing=PAD)
                Button("Clear Labels", clicked_fn=clear_labels_fn)        

    def update_class_vstack(self):
        """Clears and then rebuilds the vstack with all colors (classes) and their labels."""
        def create_class_dynamic_label(class_name, class_color):
            def change_class_color(class_name, class_label:Label, labels_labels:List[Label], rnew=None,bnew=None,gnew=None):
                r,g,b = self.voxels.get_color(class_name)
                r,g,b = r if rnew is None else rnew, g if gnew is None else gnew, b if bnew is None else bnew
                style = class_label.style
                style.update({"color":ui.color(r,g,b)})
                class_label.set_style(style)
                for label in labels_labels:
                    style = label.style
                    style.update({"color":ui.color(r,g,b)})
                    label.set_style(style)
                self.voxels.change_class_color(class_name, (r,g,b))
                # NOTE: Incantation, for some reason the colors stop updating until the stage is somehow modified.
                self.voxels.toggle()
                self.voxels.toggle()

            with HStack():
                widget = ColorWidget(width=30,style={"ColorWidget":
                                    {"border_width": 2.5,"border_color": ui.color(0.15,0.15,0.15),"border_radius": 0,"margin": 0}})
                self.set_widget_color(widget,class_color)
                class_label = Label(F"{TXTPAD}{class_name}", style={"font_size": 28.0, "color":ui.color(*class_color), "stroke":ui.color(0,0,0)})

            label_items = []
            for label in self.voxels.get_labels(class_name):
                label_items.append(Label(F"{TXTPAD}{label}", style={"font_size": 20.0, "color":ui.color(*class_color)}))

            rmodel, gmodel, bmodel, __ = [widget.model.get_item_value_model(item_model) \
                                          for item_model in widget.model.get_item_children()]  
            # NOTE: These conventions are, frankly, totally lost to me. This particular incantation works (G,B column swap)
            rmodel.add_value_changed_fn(lambda avm : change_class_color(class_name, class_label, label_items, avm.as_float, bmodel.as_float, gmodel.as_float))
            gmodel.add_value_changed_fn(lambda avm : change_class_color(class_name, class_label, label_items, rmodel.as_float, bmodel.as_float, avm.as_float))
            bmodel.add_value_changed_fn(lambda avm : change_class_color(class_name, class_label, label_items, rmodel.as_float, avm.as_float, gmodel.as_float))            

        self.class_vstack.clear()
        with self.class_vstack:
            if len(self.voxels.classes) == 0:
                # if 0 we still need a child present to force the vstack to resize
                Label(F"{TXTPAD}<no classes added>", style={"font_size": 28.0, "color":ui.color(0.675,0.675,0.675)})
                return
            for class_name in self.voxels.classes:
                class_color = self.voxels.get_color(class_name)
                create_class_dynamic_label(class_name, class_color)

    #---------------------------------------------------------------------------#
    #   this block has buttons/functions that make voxels appear in the stage   #
    #---------------------------------------------------------------------------#
    """
    """
    #CLIENT
    def send_classes(self):
        self.client.publish_class_names(self.voxels.classes)

    def request_computation(self):
        self.send_classes()
        indices = self.voxels.indices()
        voxels = self.client.request_voxel_computation()
        mask = voxels > 0 # request computation has 0 as empty and [1...N] as labels, we want [0...N-1]
        self.voxels.create_voxels(indices.view(-1,3)[mask], voxels.flatten()[mask]-1)
    #END CLIENT
    
    def build_visualization_tools(self):
        """TODO: Docs"""
        with CollapsableFrame("Voxel Visualization"):
            with VStack(height=0,spacing=PAD):
                Button("segment over labels", clicked_fn=self.request_computation)
                Button("hide/show voxels", clicked_fn=self.voxels.toggle)