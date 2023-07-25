# general python
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor
import numpy as np

# omniverse
import pxr
from pxr import Gf, UsdGeom, Sdf, Vt, UsdShade
import omni
from omni import ui

#TODO: Use @property

class Voxels:
    """Class for Visualizing Labeled Voxel Data within Omniverse. See docs for graphics.
    
    Params:
        directory (str): Where the visualizations will be created on the usd stage.
        center (float,float,float): The origin of the voxel space.
        world_dims (float,float,float): The side lengths of the volume filled by voxels.
            W (3,): ^^torch equivalent
        grid_dims (int, int, int): How many cells are along each dimension.
            G (3,): ^^torch equivalent
        
    Methods:
        toggle_visibility()
        capacity()

    """

    """
    Because OOD (out-of-domain) data is rather common, and sometimes discarding it outright is unfavorable, this class 
    is implemented to have a shell around the specified domain--OOD data gets projected onto these buffer voxels.

    grid coordinates
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━(gx,gy,gz)
    ┃ this is the shell                   ┃
    ┃    ┏━━━━━━━━━━━━━━━━━━━━━(gx-1,gy-1,┃
    ┃    ┃                         gz-1)  ┃
    ┃    ┃    user defined domain    ┃    ┃
    ┃    ┃                           ┃    ┃
    ┃ (0,0,0)━━━━━━━━━━━━━━━━━━━━━━━━┛    ┃
    ┃                                     ┃
 (-1,-1,-1)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

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

    directory : str
    "The location of the voxel prims."
    __voxel_instancer : UsdGeom.PointInstancer
    "The point instancer (UsdGeom.PointInstancer) responsible for instantiating voxels."
    __color_to_protoindex : Dict[Tuple[float,float,float],int] = {}
    "Maps (r,g,b) -> protoindex"
    __voxel_prototypes : List[str] = []
    "At each index (protoindex), the prim_path to the voxel prim corresponding to the protoindex."
    __is_visible : bool = True
    "Whether or not this extension's primitives are visible in the stage."

    # Center, World, Grid
    W:Tensor 
    "(3,) The world dimensions of the voxel space, not including the buffer shell."
    G:Tensor 
    "(3,) The grid dimensions (how many voxels along each axis)."
    _G_:Tensor 
    "(3,) The grid dimensions *including* the buffer shell (+2 each dimension). The underscores denote the buffer."
    world_dims:Tuple[float,float,float]
    "(wx, wy, wz)   The world dimensions of the voxel space, not including the buffer shell."
    grid_dims:Tuple[int,int,int]        
    "(gx, gy, gz)   The grid dimensions."
    _grid_dims_:Tuple[int,int,int]        
    "(gx+2,gy+2,gz+2)   The grid dimensions including the buffer shell."

    # Indices and Centers
    _voxel_indices_ : Tensor
    "(gx,gy,gz,:) At each i,j,k, a tensor corresponding to that voxel index."
    _voxel_centers_ : Tensor
    "(gx,gy,gz,:) At each i,j,k, a tensor corresponding to the center in world space of voxel_ijk."

    def __init__(self, grid_dims, world_dims, device='cpu', directory="/World/voxels"):
        """TODO: Docs"""
        self.device = device
        self.directory = directory
        self.grid_dims   = grid_dims
        self.world_dims  = world_dims
        self.G = torch.tensor([grid_dims[0], grid_dims[1], grid_dims[2]], device=self.device)
        self.W = torch.tensor([world_dims[0],world_dims[1],world_dims[2]], device=self.device)

        _G_ = self.G + 2 # _G_ is meant to denote the _buffer_ on each dimension.

        X = torch.arange(_G_[0], device=self.device).view(-1, 1, 1).expand( -1, _G_[1],_G_[2])
        Y = torch.arange(_G_[1], device=self.device).view( 1,-1, 1).expand(_G_[0], -1, _G_[2])
        Z = torch.arange(_G_[2], device=self.device).view( 1, 1,-1).expand(_G_[0],_G_[1], -1 )

        # Shifts everything down one in all dimensions to make the buffer.
        self._voxel_indices_ = torch.stack((X,Y,Z), dim=-1) - 1 
        self._voxel_centers_ = (1/_G_ * (self._voxel_indices_ - (_G_-1)/2)) * self.W
        self.num_voxels      = self.capacity()

    def resize_domain(self, grid_dims:Tuple[int,int,int], world_dims : Tuple[float,float,float]):
        self.__color_to_protoindex.clear()
        self.__voxel_prototypes = []

        # Initialize like normal--------------------
        self.grid_dims   = grid_dims
        self.world_dims  = world_dims
        self.G = torch.tensor([grid_dims[0], grid_dims[1], grid_dims[2]], device=self.device)
        self.W = torch.tensor([world_dims[0],world_dims[1],world_dims[2]], device=self.device)

        _G_ = self.G + 2 # _G_ is meant to denote the _buffer_ on each dimension.

        X = torch.arange(_G_[0], device=self.device).view(-1, 1, 1).expand( -1, _G_[1],_G_[2])
        Y = torch.arange(_G_[1], device=self.device).view( 1,-1, 1).expand(_G_[0], -1, _G_[2])
        Z = torch.arange(_G_[2], device=self.device).view( 1, 1,-1).expand(_G_[0],_G_[1], -1 )

        # Shifts everything down one in all dimensions to make the buffer.
        self._voxel_indices_ = torch.stack((X,Y,Z), dim=-1) - 1 
        self._voxel_centers_ = (1/_G_ * (self._voxel_indices_ - (_G_-1)/2)) * self.W
        self.num_voxels      = self.capacity()
        #-------------------------------------------

    def capacity(self, include_buffer=False) -> int:
        "Number of total possible voxels in the voxel grid."
        roving_product = 1
        for dim in self.grid_dims:
            roving_product *= dim + 0 if not include_buffer else 2
        return roving_product
    
    def toggle_visibility(self, set_visibility_setting: Optional[bool]=None):
        """Toggles or sets global visibility for *all* voxels in the stage.
        Args:
            set_visibility_setting (bool option): If None, toggles visibility, otherwise sets the visibility."""
        self.__is_visible = (not self.__is_visible) if set_visibility_setting is None else\
                          (set_visibility_setting)

        imageable = UsdGeom.Imageable(self.__voxel_instancer)

        if self.__is_visible:
            imageable.MakeVisible()
            self.__is_visible = True
        else:
            imageable.MakeInvisible()
            self.__is_visible = False

    def initialize_instancer(self):
        """TODO: Docs"""
        omni.usd.get_context().get_stage().RemovePrim(F"{self.directory}/voxel_instancer")
        voxel_instancer_prim_path = F"{self.directory}/voxel_instancer"
        stage = omni.usd.get_context().get_stage()
        self.__voxel_instancer = UsdGeom.PointInstancer.Define(stage, voxel_instancer_prim_path)

    def register_new_voxel_color(self, color : Tuple[float,float,float], invisible=False) -> int:
        """The protoindex of the voxel with the specified color.
        Args:
            color (int): The integer containing which color the voxel will be. 
            NOTE r,g,b /in [0,1].
            invisible (bool option): Whether or not this voxel should be visible.
            
        Returns: 
            protoindex (int): The protoindex pointing to the voxel prototype of the given color."""
        
        # TODO: This should be an async call in the init which waits for the stage to be initialized but this works.
        if not hasattr(self, "voxel_instancer"):
            self.initialize_instancer()  

        # If already registered just return the existing version.
        if color in self.__color_to_protoindex.keys():
            return self.__color_to_protoindex[color]
        
        stage = omni.usd.get_context().get_stage()
       
        # Create a new cube prim
        prim_name = F"voxel_{int(255*color[0])}_{int(255*color[1])}_{int(255*color[2])}" # TODO: Use class name?
        prim_path = F"{self.directory}/prototypes/voxel_{prim_name}"
        cube = UsdGeom.Cube.Define(stage, prim_path)

        # Scale to the correct voxel dimensions
        sx, sy, sz = (self.W/(2*self.G))
        xformable = UsdGeom.Xformable(cube)

        # Obtain all transformation operations on the Xformable
        ops = xformable.GetOrderedXformOps()

        # Check if a scale op already exists
        scaleOp = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scaleOp = op
                break

        # If it doesn't exist, add a new scale op
        if not scaleOp:
            scaleOp = xformable.AddScaleOp()

        # Set the value for the scale op
        scaleOp.Set(value=(sx, sy, sz))

        # Set color
        cube.GetDisplayColorAttr().Set([(Gf.Vec3f(*color))])

        # TODO: Material/Opacity

        # Add new prototype to the prototype relations (PrototypesRel)
        self.__voxel_instancer.GetPrototypesRel().AddTarget(prim_path)

        # Update backend
        new_index = len(self.__color_to_protoindex.keys())
        self.__color_to_protoindex.update({color : new_index})
        self.__voxel_prototypes.append(prim_path)

        # TODO: Figure out how to hide the prototype but not the instances.

        if invisible:
            UsdGeom.Imageable(cube).MakeInvisible()

        return new_index
    
    def get_voxel_centers(self, voxel_indices : Tensor) -> Tensor:
        """Is a (N,3) tensor of the centers of all voxels in [voxel_indices].
        Args:
            voxel_indices (N,3): Stack of voxel indices of voxel to get the center of.
        Requires:
            Forall n, voxel_indices[n,:] \in [-1...G] inclusive (per dimension of G)"""
        voxel_indices
        return self._voxel_centers_[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], :]

    def create_voxels(self, voxel_indices : Tensor, voxel_classes : Tensor):  
        """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists.
        Args:
            voxel_indices (N,3): Stack of row vectors [i,j,k] denoting which voxels to create.
            voxel_classes (N): The n-th class labels corresponds to the n-th ijk tuple in the above/parallel tensor.
        NOTE:
            This method has a suboptimal number of conversions from Tensor -> Ve"""

        # Satisfy buffer convention.
        voxel_indices += 1
    
        voxel_centers = Vt.Vec3fArray.FromNumpy(self.get_voxel_centers(voxel_indices).cpu().numpy())
        self.__voxel_instancer.CreatePositionsAttr(voxel_centers)
        self.__voxel_instancer.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(voxel_classes.cpu().numpy()))  

    def indices(self, include_shell:Optional[bool]=False):
        G = self.G
        X   = torch.arange(G[0], device=self.device).view(-1, 1, 1).expand( -1, G[1],G[2])
        Y   = torch.arange(G[1], device=self.device).view( 1,-1, 1).expand(G[0], -1, G[2])
        Z   = torch.arange(G[2], device=self.device).view( 1, 1,-1).expand(G[0],G[1], -1 )
        return torch.stack((X,Y,Z), dim=-1) 

    def shell(self):
        """TODO: Docs"""
        gx,gy,gz = self.grid_dims
        gx,gy,gz = gx+2,gy+2,gz+2

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

        return (Tensor(indices)-1).long()