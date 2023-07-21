from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor
import numpy as np
import pxr
from pxr import Gf, UsdGeom, Sdf, Vt, UsdShade
import omni
from omni import ui
import asyncio

# TODO: Use @property to deal with the managed variables in this class.
def get_stage():
    return omni.usd.get_context().get_stage()

class Voxels:
    """Class for Visualizing Voxel Data within Omniverse.
    TODO: Fill out Docs."""
    #---------------------#
    #   class variables   #
    #---------------------#
    voxel_prim_directory : str = "/World/voxseg"
    "The location of the voxel prims."

    voxel_instancer : UsdGeom.PointInstancer
    "The point instancer (UsdGeom.PointInstancer) responsible for instantiating voxels."

    color_to_protoindex : Dict[Tuple[float,float,float],int] = {}
    "Maps (r,g,b) -> protoindex"

    voxel_prototypes : List[str] = []
    "At each index (protoindex), the prim_path to the voxel prim corresponding to the protoindex."

    is_visible : bool = True
    "Whether or not this extension's primitives are visible in the stage."

    #---------------------------#
    #   voxel grid parameters   #
    #---------------------------#
    G:Tensor 
    "(3,) The grid dimensions (how many voxels along each axis)."

    grid_dims:Tuple[int,int,int]        
    "(gx, gy, gz)   The grid dimensions."

    W:Tensor 
    "(3,) The world dimensions of the voxel space, not including the buffer shell."

    world_dims:Tuple[float,float,float]
    "(wx, wy, wz)   The world dimensions of the voxel space, not including the buffer shell."

    _G_:Tensor 
    "(3,) The grid dimensions *including* the buffer shell (+2 each dimension). The underscores denote the buffer."

    _grid_dims_:Tuple[int,int,int]        
    "(gx+2,gy+2,gz+2)   The grid dimensions including the buffer shell."

    _voxel_indices_ : Tensor
    "(gx,gy,gz,:) At each i,j,k, a tensor corresponding to that voxel index."

    _voxel_centers_ : Tensor
    "(gx,gy,gz,:) At each i,j,k, a tensor corresponding to the center in world space of voxel_ijk."

    def __init__(self, world_dims, grid_dims, device='cpu', voxel_prim_directory=voxel_prim_directory):
        """TODO: Docs"""
        self.grid_dims      = grid_dims
        self._grid_dims_    = grid_dims[0]+2, grid_dims[1]+2, grid_dims[2]+2
        self.world_dims     = world_dims

        self.device         = device
        self.voxel_prim_directory = voxel_prim_directory

        G = self.G = torch.tensor([grid_dims[0], grid_dims[1], grid_dims[2]], device=self.device)
        W = self.W = torch.tensor([world_dims[0],world_dims[1],world_dims[2]], device=self.device)

        # Create buffer, a voxel shell capping each dimension. Underscores are intended to denote the boundary _GRID_.
        _G_ = self._G_ = torch.tensor([G[0]+2, G[1]+2, G[2]+2], device=self.device) 
        X   = torch.arange(_G_[0], device=self.device).view(-1, 1, 1).expand(-1,    _G_[1],_G_[2])
        Y   = torch.arange(_G_[1], device=self.device).view(1, -1, 1).expand(_G_[0],-1,    _G_[2])
        Z   = torch.arange(_G_[2], device=self.device).view(1, 1, -1).expand(_G_[0],_G_[1],-1    )

        # Shifts everything down one in all dimensions to make the buffer.
        self._voxel_indices_ = torch.stack((X,Y,Z), dim=-1) - 1 
        self._voxel_centers_ = (1/G * (self._voxel_indices_ - (G-1)/2)) * W
        self._voxel_prims_   = {}
        self.num_voxels      = _G_[0]*_G_[1]*_G_[2]

    def toggle_global_visibility(self, set_visibility_setting: Optional[bool]=None):
        """Toggles or sets global visibility for *all* voxels in the stage.
        Args:
            set_visibility_setting (bool option): If None, toggles visibility, otherwise sets the visibility."""
        self.is_visible = (not self.is_visible) if set_visibility_setting is None else\
                          (set_visibility_setting)

        imageable = UsdGeom.Imageable(self.voxel_instancer)

        if self.is_visible:
            print("Making Visible")
            imageable.MakeVisible()
            self.is_visible = True
        else:
            print("Making Invisible")
            imageable.MakeInvisible()
            self.is_visible = False

    def capacity(self, include_buffer=False) -> int:
        "Number of total possible voxels in the voxel grid."
        roving_product = 1
        for dim in (self.grid_dims if include_buffer else self._grid_dims_):
            roving_product *= dim
        return roving_product

    def initialize_instancer(self):
        """TODO: Docs"""
        voxel_instancer_prim_path = F"{self.voxel_prim_directory}/voxel_instancer"
        self.voxel_instancer = UsdGeom.PointInstancer.Define(get_stage(), voxel_instancer_prim_path)

    def register_new_voxel_color(self, color : Tuple[float,float,float], invisible=False) -> int:
        """The protoindex of the voxel with the specified color.
        Args:
            color (int): The integer containing which color the voxel will be. 
            NOTE r,g,b /in [0,1].
            
        Returns: 
            protoindex (int): The protoindex pointing to the voxel prototype of the given color."""
        
        # TODO: This should be an async call in the init which waits for the stage to be initialized.     
        
        self.initialize_instancer()  

        # If already registered just return the existing version.
        if color in self.color_to_protoindex.keys():
            return self.color_to_protoindex[color]
        
        stage = omni.usd.get_context().get_stage()
       
        # Create a new cube prim
        prim_name = F"voxel_{int(255*color[0])}_{int(255*color[1])}_{int(255*color[2])}" # TODO: Maybe take in a custom name.
        prim_path = F"{self.voxel_prim_directory}/prototypes/voxel_{prim_name}"
        cube = UsdGeom.Cube.Define(stage, prim_path)

        # Scale to the correct voxel dimensions
        sx, sy, sz = (self.W/(2*self.G))
        UsdGeom.Xformable(cube).AddScaleOp().Set(value=(sx,sy,sz))

        # Set color
        cube.GetDisplayColorAttr().Set([(Gf.Vec3f(*color))])

        # TODO: Material/Opacity

        # Add new prototype to the prototype relations (PrototypesRel)
        self.voxel_instancer.GetPrototypesRel().AddTarget(prim_path)

        # Update backend
        new_index = len(self.color_to_protoindex.keys())
        self.color_to_protoindex.update({color : new_index})
        self.voxel_prototypes.append(prim_path)

        # TODO: Figure out how to hide the prototype but not the instances. Some voxels we do want invisible though, this does that.
        if invisible:
            UsdGeom.Imageable(cube).MakeInvisible()

        return new_index
    
    def get_voxel_centers(self, voxel_indices : Tensor) -> Tensor:
        """Is a (N,3) tensor of the centers of all voxels in [voxel_indices].
        Args:
            voxel_indices (N,3): Stack of voxel indices of voxel to get the center of.
        Requires:
            Forall n, voxel_indices[n,:] \in [-1...G] inclusive (per dimension of G)"""
        voxel_indices += 1
        return self._voxel_centers_[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], :]

    def display_voxels(self, voxel_indices : Tensor, voxel_classes : Tensor):  
        """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists.
        Args:
            voxel_indices (N,3): Stack of row vectors [i,j,k] denoting which voxels to create.
            voxel_classes (N): The n-th class labels corresponds to the n-th ijk tuple in the above/parallel tensor.
        NOTE:
            This method has a suboptimal number of conversions from Tensor -> Ve"""

        # Satisfy buffer convention.
        voxel_indices += 1
    
        voxel_centers = Vt.Vec3fArray.FromNumpy(self.get_voxel_centers(voxel_indices).cpu().numpy())

        self.voxel_instancer.CreatePositionsAttr(voxel_centers)

        self.voxel_instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(voxel_classes.cpu().numpy()))  













def __DEPRECATED__create_mesh_voxel_prototype(self):
    """Creates the prototype voxel for the instancer."""
    """ 2---4                    5-------4      
        | \ |                   /|      /|      Triangle Faces (12)
    2---3---5---4---2          7-|-----6 |      342 354 375 307 310 321
    | \ | / | \ | / |   -->    | 3-----|-2      764 745 701 716 146 124
    1---0---7---6---1          |/      |/       
        | \ |                  0-------1        NOTE: These faces have outwards normal.
        1---6"""  
    unit_box = torch.tensor([(-1,-1,-1),(1,-1,-1),(1,1,-1),
                                (-1,1,-1),(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1)],device=self.device)

    vertices = (unit_box * self.W) / (2 * self.G) #(8,3)

    vec3f_list = []
    for vertex in vertices:
        vec3f_list.append(Gf.Vec3f(float(vertex[0]), float(vertex[1]), float(vertex[2])))

    mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage())
    mesh.CreatePointsAttr(vec3f_list)
    mesh.CreateFaceVertexCountsAttr(12*[3]) # 12 tris per cube, 3 vertices each
    mesh.CreateFaceVertexIndicesAttr([3,4,2,  3,5,4,  3,7,5,  3,0,7,  3,1,0,  3,2,1,
                                        7,6,4,  7,4,5,  7,0,1,  7,1,6,  1,4,6,  1,2,4])
    
#NOTE: May be soon deprecated if I find a way to make an instancer.
def __DEPRECATED__create_voxel_prims(self, voxel_indices : Tensor):  
    """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists.
    Args:
        voxel_indices (N,3): Stack of row vectors [i,j,k] denoting which voxels to create."""

    def vertices_to_vec3f(vertices_tensor : Tensor) -> List[List[Gf.Vec3f]]:
        """Converts a tensor of vertices (N,8,3) in a torch tensor to a Gf.Vec3f list list."""
        (N,_,__) = vertices_tensor.size()
        gf_vec3f_list_list = []
        for n in range(N):
            gf_vec3f_list = []
            for v in range(8):
                gf_vec3f_list.append(Gf.Vec3f(float(vertices_tensor[n,v,0].item()),
                                                float(vertices_tensor[n,v,1].item()),
                                                float(vertices_tensor[n,v,2].item())))
            gf_vec3f_list_list.append(gf_vec3f_list)
        return gf_vec3f_list_list
    
    def vertices_from_centers(centers : Tensor) -> Tensor: 
        """Is the (N,8,3) tensor containing the vertices for the center, satisfying the below convention.
        Args:
            centers (N,3): Each row is the world coordinate [wx,wy,wz] which is at the center of the cube."""
        """ 2---4                    5-------4      
            | \ |                   /|      /|      Triangle Faces (12)
        2---3---5---4---2          7-|-----6 |      342 354 375 307 310 321
        | \ | / | \ | / |   -->    | 3-----|-2      764 745 701 716 146 124
        1---0---7---6---1          |/      |/       
            | \ |                  0-------1        NOTE: These faces have outwards normal.
            1---6"""  
        unit_box = torch.tensor([(-1,-1,-1),(1,-1,-1),(1,1,-1),
                                    (-1,1,-1),(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1)],device=self.device)
        
        unit_box = unit_box.view(8, 1, 3)
        
        centers = centers.view(1, -1, 3)
        
        vertices = (unit_box * self.W) / (2 * self.G) + centers
        
        # Reshape to (N, 8, 3) for convenience
        vertices = vertices.transpose(0, 1)
        return vertices
    
    voxel_indices += 1
    (N,_) = voxel_indices.size()

    gf_vec3f_list_list = vertices_to_vec3f(vertices_from_centers(self.get_voxel_centers(voxel_indices)))

    for n in range(N):
        voxel_prim_path = \
        F"{self.voxel_prim_directory}/voxel_{voxel_indices[n][0]}_{voxel_indices[n][1]}_{voxel_indices[n][2]}"

        self._voxel_prims_[voxel_indices[n]] = voxel_prim_path   

        mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), voxel_prim_path)
        mesh.CreatePointsAttr(gf_vec3f_list_list[n])
        mesh.CreateFaceVertexCountsAttr(12*[3]) # 12 tris per cube, 3 vertices each
        mesh.CreateFaceVertexIndicesAttr([3,4,2,  3,5,4,  3,7,5,  3,0,7,  3,1,0,  3,2,1,
                                        7,6,4,  7,4,5,  7,0,1,  7,1,6,  1,4,6,  1,2,4,])