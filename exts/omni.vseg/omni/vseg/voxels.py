from typing import Callable, List, Sequence, Tuple
import torch
from torch import Tensor
import pxr
from pxr import Gf, UsdGeom, Sdf
import omni
import numpy as np

class Voxels():
    """Visualizes voxels."""
    grid_dims       : Tuple[int,int,int]        
    "(gx, gy, gz) The grid dimensions."
    full_grid_dims  : Tuple[int,int,int]        
    "(gx+2,gy+2,gz+2) The grid dimensions including the buffer shell."
    world_dims      : Tuple[float,float,float]
    "(wx, wy, wz) The world dimensions of the voxel space, not including the buffer shell."

    G : Tensor 
    "(3,)=[gx, gy, gz] The grid dimensions (how many voxels along each axis)."
    _G_ : Tensor 
    "(3,)=[gx+2,gy+2,gz+2] The grid dimensions *including* the buffer shell. The underscores denote the buffer."
    W : Tensor 
    "(3,)=[wx, wy, wz] The world dimensions of the voxel space, not including the buffer shell."

    __voxel_indices : Tensor
    "(gx,gy,gz,:)=[i,j,k] At each i,j,k, a tensor corresponding to that voxel index."
    __voxel_centers : Tensor
    "(gx,gy,gz,:)=[wx,wy,wz] At each i,j,k, a tensor corresponding to the center of voxel_ijk."
    voxels        : dict 
    "Maps voxel index (i,j,k) to the primpath of the voxel prim."
    num_voxels    : int
    "Number of total possible voxels in the voxel grid."

    def __init__(self, world_dims, grid_dims, device='cuda'):

        self.grid_dims = grid_dims
        self.full_grid_dims = (grid_dims[0]+2, grid_dims[1]+2, grid_dims[2]+2)
        self.world_dims = world_dims

        G = self.G = torch.tensor([grid_dims[0], grid_dims[1], grid_dims[2]], device=device)
        W = self.W = torch.tensor([world_dims[0],world_dims[1],world_dims[2]], device=device)
        # Includes buffer, a layer of voxels along each boundary. NOTE: Underscores illustrate the boundary _GRID_.
        _G_ = self._G_ = torch.tensor([G[0]+2, G[1]+2, G[2]+2], device=device) 
        X = torch.arange(_G_[0], device=device).view(-1, 1, 1).expand(-1,      _G_[1], _G_[2])
        Y = torch.arange(_G_[1], device=device).view(1, -1, 1).expand(_G_[0],  -1,     _G_[2])
        Z = torch.arange(_G_[2], device=device).view(1, 1, -1).expand(_G_[0],  _G_[1], -1    )

        # Shifts everything down one in all dimensions to make the buffer.
        self.__voxel_indices = torch.stack((X,Y,Z), dim=-1) - 1 
        self.__voxel_centers = (1/G * (self.__voxel_indices - (G-1)/2)) * W

        self.voxels = {}
        self.num_voxels = _G_[0]*_G_[1]*_G_[2]

    def get_voxel_index(self, voxel_index_ijk : Tuple[int,int,int]) -> Tuple[int,int,int]:
        """The index to find the i,j,k-th voxel at.

        Requires:
            i,j,k \in [-1...G] inclusive (per dimension of G)"""
        i,j,k = voxel_index_ijk
        return (i+1,j+1,k+1)
    
    def get_voxel_center(self, voxel_index_ijk : Tuple[int,int,int]) -> Tensor:
        """The center of the i,j,k-th voxel.

        Requires:
            i,j,k \in [-1...G] inclusive (per dimension of G)"""
        i,j,k = voxel_index_ijk
        return self.__voxel_centers[i+1,j+1,k+1]

    #NOTE: May be deprecated in favor of methods which just spawn via cube and transform.
    def create_voxel_prim(self, voxel_index_ijk : Tuple[int,int,int], device='cuda') -> str:  
        """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists."""
        i,j,k = voxel_index_ijk[:]
        i,j,k = i+1,j+1,k+1
        
        voxel_prim_path = F"/World/Voxels/voxel_{i}_{j}_{k}"
        if voxel_index_ijk in self.voxels:
            apply_color_to_prim((np.random.random(),np.random.random(),np.random.random()))(voxel_prim_path)
            return voxel_prim_path
        
        self.voxels[voxel_index_ijk]=voxel_prim_path
        
        def Gf_Vec3f_list_of_vertices_tensor(vertices_tensor) -> List[Gf.Vec3f]:
            """Converts a tensor of vertices in a torch tensor to a list of Gf.Vec3f tensors."""
            gfvec3f_list = []
            for vertex in vertices_tensor:
                gfvec3f_list.append(Gf.Vec3f(float(vertex[0]), float(vertex[1]), float(vertex[2])))
            return gfvec3f_list
        
        def get_vertices_of_center(center : Tensor) -> Tensor: 
            """Is the set of vertices which make up the voxel for the voxel at [voxel_index_ijk]."""
            """ 2---4                    5-------4      
                | \ |                   /|      /|      Triangle Faces (12)
            2---3---5---4---2          7-|-----6 |      342 354 375 307 310 321
            | \ | / | \ | / |   -->    | 3-----|-2      764 745 701 716 146 124
            1---0---7---6---1          |/      |/       
                | \ |                  0-------1        NOTE: These faces have outwards normal
                1---6"""  
            unit_box = torch.tensor([(-1,-1,-1),(1,-1,-1),(1,1,-1),
                                     (-1,1,-1),(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1)],device=device)
            return (unit_box*self.W)/(2*self.G) + center
        
        
        mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), voxel_prim_path)
        mesh.CreatePointsAttr(Gf_Vec3f_list_of_vertices_tensor(get_vertices_of_center(self.__voxel_centers[i,j,k])))
        mesh.CreateFaceVertexCountsAttr(12*[3]) # 12 tris per cube, 3 vertices each
        mesh.CreateFaceVertexIndicesAttr([3,4,2,  3,5,4,  3,7,5,  3,0,7,  3,1,0,  3,2,1,
                                          7,6,4,  7,4,5,  7,0,1,  7,1,6,  1,4,6,  1,2,4,])
        
        apply_color_to_prim((np.random.random(),np.random.random(),np.random.random()))(voxel_prim_path)        
                                          
        return voxel_prim_path

def apply_color_to_prim(color: tuple) -> Callable[[str],None]:
    def __apply_color_to_prim_helper(prim_path : str) -> Tuple[float,float,float] :
        """
        Apply an RGB color to a prim. 

        Parameters:
        prim_path (str): The path to the prim.
        color (tuple): The RGB color to apply as a tuple of three floats.
        """
        stage = omni.usd.get_context().get_stage()

        # Access the prim
        prim = stage.GetPrimAtPath(prim_path)

        # Check if the prim exists
        if not prim:
            print(f'Error: No prim at {prim_path}')
            return

        # Get the UsdGeom interface for the prim
        prim_geom = pxr.UsdGeom.Gprim(prim)

        # Create a color attribute if it doesn't exist
        if not prim_geom.GetDisplayColorAttr().HasAuthoredValueOpinion():
            prim_geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        # Set the color
        else:
            prim_geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return __apply_color_to_prim_helper