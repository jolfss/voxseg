# general python
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor
import numpy as np

# omniverse
from pxr import Gf, UsdGeom, Vt
import omni

#TODO: Use @property

class Voxels:
    """Class for Visualizing Labeled Voxel Data within Omniverse. See init docs for graphics.
    
    Params:
        directory (str): Where the visualizations will be created on the usd stage.
        center (float,float,float): The origin of the voxel space.
        world_dims (float,float,float): The side lengths of the volume filled by voxels.
            W (3,): ^^torch equivalent
        grid_dims (int, int, int): How many cells are along each dimension.
            G (3,): ^^torch equivalent
        
    Methods:
        toggle(force_visibility:bool option=None): Sets the visibility of all of the voxels in the stage. 
        capacity(include_shell:bool=False) -> 

    """
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

    directory : str
    "The location of the voxel prims."
    _voxel_instancer : UsdGeom.PointInstancer
    "The point instancer (UsdGeom.PointInstancer) responsible for instantiating voxels."
    _color_to_protoindex : Dict[Tuple[float,float,float],int] = {}
    "Maps (r,g,b) -> protoindex"
    _voxel_prototypes : List[str] = []
    "At each index (protoindex), the prim_path to the voxel prim corresponding to the protoindex."
    _is_visible : bool = True
    "Whether or not this extension's primitives are visible in the stage."

    # Grid, World
    G:Tensor 
    "(3,) The grid dimensions (how many voxels along each axis)."
    W:Tensor 
    "(3,) The world dimensions of the voxel space, not including the buffer shell."
    grid_dims:Tuple[int,int,int]        
    "(gx, gy, gz)   The grid dimensions."
    world_dims:Tuple[float,float,float]
    "(wx, wy, wz)   The world dimensions of the voxel space, not including the buffer shell."
    
    def stage(self):
        try:
            return omni.usd.get_context().get_stage()
        except:
            print("[voxseg] Warning: Stage has not yet been initialized (did you try to call me in __init__?)")
            return None

    def __init__(self, grid_dims, world_dims, device='cpu', directory="/World/voxvis"):
        """TODO: Docs"""
        self.device = device
        self.directory = directory
        self.grid_dims   = grid_dims
        self.world_dims  = world_dims
        self.G = torch.tensor([grid_dims[0], grid_dims[1], grid_dims[2]], device=self.device)
        self.W = torch.tensor([world_dims[0],world_dims[1],world_dims[2]], device=self.device)

    def __initialize_instancer(self):
        """TODO: This should be an async event which is triggered by the stage spawning.
        The extension loads before the stage so it cannot be done during __init__."""
        omni.usd.get_context().get_stage().RemovePrim(F"{self.directory}/voxel_instancer")
        voxel_instancer_prim_path = F"{self.directory}/voxel_instancer"
        stage = self.stage()
        self._voxel_instancer = UsdGeom.PointInstancer.Define(stage, voxel_instancer_prim_path)

    #Getters
    def indices(self, include_buffer:bool=False):
        """TODO: Docs"""
        G = self.G + 2 if include_buffer else self.G
        X = torch.arange(G[0], device=self.device).view(-1, 1, 1).expand( -1, G[1],G[2])
        Y = torch.arange(G[1], device=self.device).view( 1,-1, 1).expand(G[0], -1, G[2])
        Z = torch.arange(G[2], device=self.device).view( 1, 1,-1).expand(G[0],G[1], -1 )
        return torch.stack((X,Y,Z), dim=-1).long() - (1 if include_buffer else 0)

    def indices_shell(self, include_buffer:bool=False):
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

    def capacity(self, include_buffer:bool=False) -> int:
        "Number of total possible voxels in the voxel grid."
        roving_product = 1
        for dim in self.grid_dims:
            roving_product *= dim + 0 if not include_buffer else 2
        return roving_product

    def centers(self, indices:Tensor):
        """The world-coordinate centers of all voxels in [indices]. Does *NOT* need to be in the span of the voxel grid
        Args:
            indices (*,3): Any size with last component (,3) representing ijk indices for the voxel."""
        if indices.type() != 'torch.LongTensor':
            print("[voxvis] Warning: Centers was passed non-long-typed indices, may not be aligned with true centers.")
        return (1/(self.G) * (indices - (self.G-1)/2)) * self.W
    
    def toggle(self, force_visibility:Optional[bool]=None):
        """Toggles or sets global visibility for *all* voxels in the stage.
        Args:
            force (bool option): Sets the visibility regardless of current state."""
        self._is_visible = (not self._is_visible) if force_visibility is None else (force_visibility)

        imageable = UsdGeom.Imageable(self._voxel_instancer)
        imageable.MakeVisible() if self._is_visible else imageable.MakeInvisible()

    def register_new_voxel_color(self, color : Tuple[float,float,float], invisible=False) -> int:
        """The protoindex of the voxel with the specified color.
        Args:
            color (float,float,float): The rgb tuple containing which color the voxel will be. 
                NOTE: r,g,b /in [0,1].
            invisible (bool=False): Whether or not this voxel should be made invisible.
            
        Returns: 
            protoindex (int): The protoindex pointing to the voxel prototype of the given color."""
        
        if not hasattr(self, "_voxel_instancer"):
            self.__initialize_instancer()  
            # TODO: See __initialize_instancer docs.

        # If already registered just return the existing version.
        if color in self._color_to_protoindex.keys():
            return self._color_to_protoindex[color]
        
        stage = self.stage()
       
        # Create a new cube prim
        prim_name = F"{int(255*color[0])}_{int(255*color[1])}_{int(255*color[2])}" # TODO: Use class name?
        prim_path = F"{self.directory}/prototypes/voxel_{prim_name}"
        cube = UsdGeom.Cube.Define(stage, prim_path)

        # Obtain all transformation operations on the Xformable
        xformable = UsdGeom.Xformable(cube)
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

        # Scale to the correct voxel dimensions
        sx, sy, sz = (self.W/(2*self.G))
        scaleOp.Set(value=(sx, sy, sz))

        cube.GetDisplayColorAttr().Set([(Gf.Vec3f(*color))])

        # Add new prototype to the prototype relations (PrototypesRel)
        self._voxel_instancer.GetPrototypesRel().AddTarget(prim_path)

        # Update backend
        new_index = len(self._color_to_protoindex.keys())
        self._color_to_protoindex.update({color : new_index})
        self._voxel_prototypes.append(prim_path)

        # TODO: Figure out how to hide the prototype but not the instances.

        if invisible:
            UsdGeom.Imageable(cube).MakeInvisible()

        return new_index

    def create_voxels(self, voxel_indices : Tensor, voxel_classes : Tensor):  
        """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists.
        Args:
            voxel_indices (*,3): Some size of N total vectors [i,j,k] denoting which voxels to create.
            voxel_classes (*): Any tensor st. the n-th class (flattened) corresponds to the n-th ijk vector above.
        Requires:
            voxel_indices.numel() == voxel_classes.numel()"""
        
        # Check registration of classes
        num_classes = len(self._voxel_prototypes)
        max_id = voxel_classes.max()
        if num_classes <= max_id:
            print(F"[voxvis] Error: Cannot create voxel class {max_id}, only {num_classes} are registered. Aborting")
            return
        
        # Assign instances to instancer
        voxel_indices = voxel_indices.view(-1,3)
        voxel_centers = Vt.Vec3fArray.FromNumpy(self.centers(voxel_indices).cpu().numpy())
        self._voxel_instancer.CreatePositionsAttr(voxel_centers)
        self._voxel_instancer.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(voxel_classes.view(-1).cpu().numpy()))  


    def reinit(self, grid_dims, world_dims):
        """TODO: Docs"""
        self._color_to_protoindex.clear()
        self._voxel_prototypes = []
        self.grid_dims   = grid_dims
        self.world_dims  = world_dims
        self.G = torch.tensor([grid_dims[0], grid_dims[1], grid_dims[2]], device=self.device)
        self.W = torch.tensor([world_dims[0],world_dims[1],world_dims[2]], device=self.device)