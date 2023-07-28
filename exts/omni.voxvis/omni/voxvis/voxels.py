# general python imports
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor

# omniverse
import omni
from pxr import UsdGeom, Gf, Vt, Usd

Color = Tuple[float,float,float]
"(R,G,B) s.t. R,G,B \in [0,1]"
Protoindex = int
"""Indexes a prototype voxel (a primitive in the stage) among those currently added to the instancer's PrototypesRel.
    NOTE: Here especially, it refers to PrototypesRel.GetTargets()[protoindex]"""
Primpath = str
"The path of a primitive in the current USD."

def stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()

def warn(msg):
    print(F"[voxseg.voxel]: Warning: {msg}")

def track(func):
    def __track(self, *args, **kwargs):
        print("#━━━━━━━━━vvvv━━━━━━━━━#")
        print(F"[voxseg.track] __BEFORE__ @{self.directory}")
        self.print_state()
        func(self,*args,**kwargs)
        print("#----------------------#")
        print("[voxseg.track] __AFTER__")
        self.print_state()
        print("#━━━━━━━━━━━━━━━━━━━━━━#")
    return __track

"""Because OOD (out-of-domain) data is rather common, and sometimes discarding it outright is unfavorable, this class
    is implemented to have a shell around the specified domain--OOD data gets projected onto these buffer voxels.

    grid coordinates
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━(gx,gy,gz)
    ┃ this is the buffer                  ┃
    ┃    ┏━━━━━━━━━━━━━━━━━━━━━(gx-1,gy-1,┃
    ┃    ┃                         gz-1)  ┃
    ┃    ┃    user defined domain    ┃    ┃
    ┃    ┃                           ┃    ┃
    ┃ (0,0,0)━━━━━━━━━━━━━━━━━━━━━━━━┛    ┃
    ┃                                     ┃
(-1,-1,-1)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    world coordinates:
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━(+1 voxel)
    ┃                                     ┃
    ┃    ┏━━━━━━━━━━━━━┳━━━━━━━(wx/2,wy/2,┃
    ┃    ┃             ╹           wz/2)  ┃
    ┃    ┣━━━━━━━━ (cx,cy,cz) ━━━━━━━┫    ┃         
    ┃(-wx/2,           ╻             ┃    ┃
    ┃-wy/2,-wz/2)━━━━━━┻━━━━━━━━━━━━━┛    ┃
    ┃                                     ┃
(-1 voxel)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"""

class Voxels():
    """TODO: Docs"""

    def __init__(self, 
                 world_dims : Tuple[float,float,float], 
                 grid_dims : Tuple[int,int,int], 
                 voxel_center : Tuple[float,float,float]=(0.,0.,0.), 
                 directory : str="/World/voxvis"):
        """TODO: Docs"""

        self._world_dims = world_dims
        self._grid_dims = grid_dims
        self._voxel_center = voxel_center
        self.directory = directory
        
        self._is_visible = True
        
        self._instancer = None 

        self._class_dict : Dict[str, Tuple[List[str], Color, Primpath]] = {}
        
    #----------------#
    #   properties   #
    #----------------#
    @property
    def world_dims(self):
        return self._world_dims
    @world_dims.setter
    def world_dims(self, value):
        self._world_dims = (float(value[0]), float(value[1]), float(value[2]))
    @property
    def W(self):
        return torch.tensor(self.world_dims)
    
    @property
    def grid_dims(self):
        return self._grid_dims
    @grid_dims.setter
    def grid_dims(self, value):
        assert type(value[0]) == int and type(value[0]) == int and type(value[0]) == int
        self._grid_dims = (value[0],value[1],value[2])
    @property
    def G(self):
        return torch.tensor(self.grid_dims)
    
    @property
    def voxel_center(self):
        return self._voxel_center
    @voxel_center.setter
    def voxel_center(self, value):
        self._voxel_center = (float(value[0]), float(value[1]), float(value[2]))
    @property
    def C(self):
        return torch.tensor(self.voxel_center)

    @property
    def classes(self):
        return list(self._class_dict.keys())

    @property
    def colors(self) -> List[Color]:
        colors = []
        for class_name in self.classes:
            colors.append(self.get_color(class_name))
        return colors
    
    @property
    def prototypes(self) -> List[Primpath]:
        "The list of "
        prototypes_list = []
        for class_name in self.classes:
            prototypes_list.append(self.get_prototype(class_name))
        return prototypes_list

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def instancer(self) -> UsdGeom.PointInstancer:
        if self._instancer is None:
            stage().RemovePrim(F"{self.directory}/instancer")
            self._instancer = UsdGeom.PointInstancer.Define(stage(), F"{self.directory}/instancer")
            self._instancer.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(0*self.G.cpu().numpy()))
            self._instancer.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy((0*self.G[0]).view(-1).cpu().numpy()))  
        return self._instancer

    #-------------#
    #   getters   #
    #-------------#
    def get_labels(self, class_name : str) -> List[str]:
        return self._class_dict[class_name][0]

    def get_color(self, class_name : str) -> Color:
        return self._class_dict[class_name][1]
    
    def get_prototype(self, class_name : str) -> Primpath:
        return self._class_dict[class_name][2]

    def capacity(self, include_buffer:bool=False) -> int:
        "Number of total possible voxels in the voxel grid."
        roving_product = 1
        for dim in self.grid_dims:
            roving_product *= dim + (0 if not include_buffer else 2)
        return roving_product

    def centers(self, indices:Tensor):
        """The world-coordinate centers of all voxels in [indices]. Does *NOT* need to be in the span of the voxel grid
        Args:
            indices (*,3): Any size with last component (,3) representing ijk indices for the voxel."""
        if indices.type() != 'torch.LongTensor':
            warn("Centers was passed non-long-typed indices, may not be aligned with true centers.")
        return (1/(self.G) * (indices - (self.G-1)/2)) * self.W

    def indices(self, include_buffer:bool=False) -> Tensor:
        """TODO: Docs"""
        G = self.G + 2 if include_buffer else self.G
        X = torch.arange(G[0]).view(-1, 1, 1).expand( -1, G[1],G[2])
        Y = torch.arange(G[1]).view( 1,-1, 1).expand(G[0], -1, G[2])
        Z = torch.arange(G[2]).view( 1, 1,-1).expand(G[0],G[1], -1 )
        return torch.stack((X,Y,Z), dim=-1).long() - (1 if include_buffer else 0)

    def shell(self, include_buffer:bool=False) -> Tensor:
        """TODO: Docs"""
        gx,gy,gz = self.grid_dims
        gx,gy,gz = (gx+2,gy+2,gz+2) if include_buffer else (gx, gy, gz)

        indices = []

        for x in [0,gx-1]:
            for y in range(gy):
                for z in range(gz):
                    indices.append([x,y,z])
        for y in [0,gy-1]:
            for x in range(gx-2):
                for z in range(gz):
                    indices.append([x+1,y,z])
        for z in [0,gz-1]:
            for x in range(gx-2):
                for y in range(gy-2):
                    indices.append([x+1,y+1,z])

        return (Tensor(indices)-1).long() if include_buffer else (Tensor(indices)).long()

    #--------------#
    #   mutators   #
    #--------------#
    def add_label(self, class_name : str, label : str):
        "Adds a label to the class's labels, or does nothing if already present."
        if label not in self.get_labels(class_name):
            self.get_labels(class_name).append(label)

    def scale_update(self):
        for voxel_prim_path in self.prototypes:
            voxel_prim = stage().GetPrimAtPath(voxel_prim_path)
            xformable = UsdGeom.Xformable(voxel_prim)
            scaleOp = None # check if prim already has scale op
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scaleOp = op
                    break
            if not scaleOp:
                scaleOp = xformable.AddScaleOp()
            sx, sy, sz = (self.W/(2*self.G))
            scaleOp.Set(value=(sx, sy, sz))

    def create_class(self, class_name : str, class_color : Color):
        """Creates a new class if class_name is not in use, otherwise does nothing."""

        if class_name in self.classes or class_color in self.colors:
            return

        # create prototype voxel
        prim_name = F"{int(class_color[0]*255)}_{int(class_color[1]*255)}_{int(class_color[2]*255)}"
        prim_path : Primpath = F"{self.directory}/prototypes/voxel_{prim_name}"
        voxel_prim = UsdGeom.Cube.Define(stage(), prim_path)

        # set voxel color
        voxel_prim.GetDisplayColorAttr().Set([(Gf.Vec3f(*class_color))])

        # add new prototype to the prototype relations
        self.instancer.CreatePrototypesRel().AddTarget(prim_path)

        # update dictionary
        self._class_dict.update({class_name : ([], class_color, prim_path)})

        # set voxel scale (reads from the dictionary so needs to be called after the above)
        self.scale_update()

    def remove_label(self, class_name : str, label : str):
        """Removes the label from the class, or has no effect if not already present."""
        if label in self.get_labels(class_name):
            self.get_labels(class_name).remove(label)

    def delete_class(self, class_name : str):
        """Deletes a class and its protovoxel if they exist."""
        if class_name in self.classes:
            _,_, protovoxel_primpath = self._class_dict.pop(class_name)
            stage().RemovePrim(protovoxel_primpath)
            self._instancer = None

    def clear_classes(self, override_dict:Optional[Dict[str,Tuple[List[str],Tuple[float,float,float],Primpath]]]=None):
        for class_name in self.classes:
            self.delete_class(class_name)
        self._class_dict = {} if override_dict is None else override_dict
        
        self.instancer 
        # NOTE: Calling self.instancer is *NOT* necessary, but it forces self._instancer to reinitialize immediately

    def redomain(self,
                 world_dims:Optional[Tuple[float,float,float]]=None, 
                 grid_dims:Optional[Tuple[int,int,int]]=None, 
                 voxel_center:Optional[Tuple[float,float,float]]=None):
        self.world_dims   = world_dims   if world_dims   is not None else self.world_dims
        self.grid_dims    = grid_dims    if grid_dims    is not None else self.grid_dims
        self.voxel_center = voxel_center if voxel_center is not None else self.voxel_center
        self.scale_update()

    def toggle(self, force_visibility:Optional[bool]=None):
        """Toggles or sets global visibility for *all* voxels in the stage.
        Args:
            force (bool option): Sets the visibility regardless of current state."""
        self._is_visible = (not self._is_visible) if force_visibility is None else (force_visibility)

        imageable = UsdGeom.Imageable(self.instancer)
        imageable.MakeVisible() if self._is_visible else imageable.MakeInvisible()

    def create_voxels(self, voxel_indices : Tensor, voxel_classes : Tensor):  
        """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists.
        Args:
            voxel_indices (*,3): Some size of N total vectors [i,j,k] denoting which voxels to create.
            voxel_classes (*): Any tensor s.t. the n-th class (flattened) corresponds to the n-th ijk vector above.
        Requires:
            voxel_classes contains integers \in [0,len(self.classes)-1]
            voxel_indices.numel()/3 == voxel_classes.numel()"""
        
        # assert voxel_classes.max() < len(self.classes)
        # assert voxel_classes.min() >= 0
        # assert voxel_indices.numel()/3 == voxel_classes.numel()
        
        # Assign instances to instancer
        voxel_indices = voxel_indices.view(-1,3)
        voxel_centers = Vt.Vec3fArray.FromNumpy(self.centers(voxel_indices).cpu().numpy())
        self.instancer.CreatePositionsAttr(voxel_centers)
        self.instancer.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(voxel_classes.view(-1).cpu().numpy()))  

    #-----------#
    #   debug   #
    #-----------#
    def print_state(self):
        print("Dict:")
        print(F"├─ Classes: {self.classes}")
        print(F"├─ Color: {self.colors}")
        print(F"╰─ Prototype: {self.prototypes}")

        print("\nInstancer:",self._instancer)
        if self._instancer is not None:
            print(F"├─ Targets:")
            print(F"│\t├─ len: {len(self._instancer.GetPrototypesRel().GetTargets())}")
            print(F"│\t╰─ targets: {len(self._instancer.GetPrototypesRel().GetTargets())}")
            for i in range(len(self._instancer.GetPrototypesRel().GetTargets())):
                print(F"│\t\t├─ {self._instancer.GetPrototypesRel().GetTargets()[i]}") \
                if not i == len(self._instancer.GetPrototypesRel().GetTargets()) - 1 else \
                print(F"│\t\t╰─ {self._instancer.GetPrototypesRel().GetTargets()[-1]}")

            print(F"├─ PositionsAttr: {self._instancer.GetPositionsAttr().Get()[:3]}")
            print(F"│\t╰─ len: {len(self._instancer.GetPositionsAttr().Get())}")

            print(F"╰─ ProtoIndices: {self._instancer.GetProtoIndicesAttr().Get()[:5]}")
            print(F"\t├─ len: {len(self._instancer.GetProtoIndicesAttr().Get())}")
            print(F"\t├─ max: {min(list(self._instancer.GetProtoIndicesAttr().Get()))}")
            print(F"\t╰─ min: {max(list(self._instancer.GetProtoIndicesAttr().Get()))}")