from typing import Callable, Dict, List, Sequence, Tuple
import torch
from torch import Tensor
import pxr
from pxr import Gf, UsdGeom, Sdf, Vt
import omni

class Voxels():
    """Visualizes voxels.
    TODO: Docs
    """

    voxel_prim_directory : str
    "The location of the voxel prims. Default: /World/VOXSEG/voxels"

    G   : Tensor 
    "(3,) The grid dimensions (how many voxels along each axis)."
    grid_dims       : Tuple[int,int,int]        
    "(gx, gy, gz)   The grid dimensions."

    _G_ : Tensor 
    "(3,) The grid dimensions *including* the buffer shell (+2 each dimension). The underscores denote the buffer."
    _grid_dims_  : Tuple[int,int,int]        
    "(gx+2,gy+2,gz+2)   The grid dimensions including the buffer shell."

    world_dims      : Tuple[float,float,float]
    "(wx, wy, wz)   The world dimensions of the voxel space, not including the buffer shell."
    W   : Tensor 
    "(3,) The world dimensions of the voxel space, not including the buffer shell."

    __voxel_indices : Tensor
    "(gx,gy,gz,:) At each i,j,k, a tensor corresponding to that voxel index."
    __voxel_centers : Tensor
    "(gx,gy,gz,:) At each i,j,k, a tensor corresponding to the center in world space of voxel_ijk."
    __voxel_counts : dict = {}
    "Tensor([gx,gy,gz]) -> int, At each i,j,k, the number of times the voxel has been observed."
    __voxel_prims : dict 
    "Tensor([gx,gy,gz]) -> prim_path_ijk to the primpath of the voxel prim."

    def __init__(self, world_dims, grid_dims, device='cpu', voxel_prim_directory="/World/VOXSEG/voxels"):
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
        self.__voxel_indices = torch.stack((X,Y,Z), dim=-1) - 1 
        self.__voxel_centers = (1/G * (self.__voxel_indices - (G-1)/2)) * W
        self.__voxel_prims   = {}
        self.num_voxels      = _G_[0]*_G_[1]*_G_[2]

    def initialize_voxel_instancer(self, stage):
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

        proto_voxel_prim_path       = F"{self.voxel_prim_directory}/proto_voxel"
        voxel_instancer_prim_path   = F"{self.voxel_prim_directory}/voxel_instancer"

        mesh = UsdGeom.Mesh.Define(stage, proto_voxel_prim_path)
        mesh.CreatePointsAttr(vec3f_list)
        mesh.CreateFaceVertexCountsAttr(12*[3]) # 12 tris per cube, 3 vertices each
        mesh.CreateFaceVertexIndicesAttr([3,4,2,  3,5,4,  3,7,5,  3,0,7,  3,1,0,  3,2,1,
                                        7,6,4,  7,4,5,  7,0,1,  7,1,6,  1,4,6,  1,2,4,])
        
        self.proto_voxel_prim = stage.GetPrimAtPath(proto_voxel_prim_path)

        self.voxel_instancer = UsdGeom.PointInstancer.Define(stage, voxel_instancer_prim_path)

        # Perform David's Bizarre Incantation
        points_W = torch.zeros((self.capacity(),3), device="cuda")
        points_W = points_W.view(-1, 3)
        self.voxel_instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(points_W.cpu().numpy()))
        self.voxel_instancer.GetProtoIndicesAttr().Set([0] * points_W.shape[0])
        # End of Bizarre Incantation

        self.voxel_instancer.CreateProtoIndicesAttr([0])

        proto_rel = self.voxel_instancer.GetPrototypesRel()
        proto_rel.AddTarget(proto_voxel_prim_path)
        

    def capacity(self, include_buffer=False) -> int:
        "Number of total possible voxels in the voxel grid."
        roving_product = 1
        for dim in (self._G_ if include_buffer else self.G):
            roving_product *= dim
        return roving_product
    
    def get_voxel_centers(self, voxel_indices : Tensor) -> Tensor:
        """Is a (N,3) tensor of the centers of all voxels in [voxel_indices].

        Args:
            voxel_indices (N,3): Stack of voxel indices of voxel to get the center of.

        Requires:
            Forall n, voxel_indices[n,:] \in [-1...G] inclusive (per dimension of G)"""
        voxel_indices += 1
        return self.__voxel_centers[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], :]

    def create_voxel_prims_with_instancer(self, voxel_indices : Tensor):  
        """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists.
        Args:
            voxel_indices (N,3): Stack of row vectors [i,j,k] denoting which voxels to create."""
        
        self.initialize_voxel_instancer(omni.usd.get_context().get_stage())

        print("Instancer check START")

        voxel_indices += 1
 
        voxel_centers = Vt.Vec3fArray.FromNumpy(self.get_voxel_centers(voxel_indices).cpu().numpy())

        self.voxel_instancer.CreatePositionsAttr(voxel_centers)

        num_instances = len(voxel_indices)
        self.voxel_instancer.GetProtoIndicesAttr().Set([0] * num_instances)

        print("Instancer check COMPLETE")
  
        
    def receive_voxel_counts(self, tensor_ijk_count : Tensor):
        """Updates/populates [self.__voxel_counts] to have the count numbers for each voxel_ijk in [tensor_ijk_count].
        Args:
            tensor_ijk_count (N,4): Each row is of the form [i,j,k,count]."""
        (N,_) = tensor_ijk_count.size()
        for n in range(N):
            self.__voxel_counts.update({tensor_ijk_count[n,:3],tensor_ijk_count[3]})

    # NON INSTANCER PRIM INITIALIZATION

    #NOTE: May be soon deprecated if I find a way to make an instancer.
    def create_voxel_prims(self, voxel_indices : Tensor):  
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

            self.__voxel_prims[voxel_indices[n]] = voxel_prim_path   

            mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), voxel_prim_path)
            mesh.CreatePointsAttr(gf_vec3f_list_list[n])
            mesh.CreateFaceVertexCountsAttr(12*[3]) # 12 tris per cube, 3 vertices each
            mesh.CreateFaceVertexIndicesAttr([3,4,2,  3,5,4,  3,7,5,  3,0,7,  3,1,0,  3,2,1,
                                            7,6,4,  7,4,5,  7,0,1,  7,1,6,  1,4,6,  1,2,4,])
    
    # DEPRECATED FOR MULTIPLICITY VERSION
    # #NOTE: May be deprecated in favor of methods which just spawn via cube and transform.
    # def create_voxel_prim(self, voxel_index_ijk : Tuple[int,int,int]) -> str:  
    #     """Creates the voxel for the i,j,k-th voxel in the stage, or does nothing if it already exists."""
    #     i,j,k = voxel_index_ijk[:]
    #     i,j,k = i+1,j+1,k+1
        
    #     voxel_prim_path = F"{self.voxel_prim_directory}/voxel_{i}_{j}_{k}"

    #     self.__voxel_prims[voxel_index_ijk]=voxel_prim_path
        
    #     def Gf_Vec3f_list_of_vertices_tensor(vertices_tensor) -> List[Gf.Vec3f]:
    #         """Converts a tensor of vertices in a torch tensor to a list of Gf.Vec3f tensors."""
    #         gfvec3f_list = []
    #         for vertex in vertices_tensor:
    #             gfvec3f_list.append(Gf.Vec3f(float(vertex[0]), float(vertex[1]), float(vertex[2])))
    #         return gfvec3f_list
        
    #     def vertices_from_centers(center : Tensor) -> Tensor: 
    #         """Is the set of vertices which make up the voxel for the voxel at [voxel_index_ijk]."""
    #         """ 2---4                    5-------4      
    #             | \ |                   /|      /|      Triangle Faces (12)
    #         2---3---5---4---2          7-|-----6 |      342 354 375 307 310 321
    #         | \ | / | \ | / |   -->    | 3-----|-2      764 745 701 716 146 124
    #         1---0---7---6---1          |/      |/       
    #             | \ |                  0-------1        NOTE: These faces have outwards normal
    #             1---6"""  
    #         unit_box = torch.tensor([(-1,-1,-1),(1,-1,-1),(1,1,-1),
    #                                  (-1,1,-1),(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1)],device=self.device)
    #         return (unit_box*self.W)/(2*self.G) + center
        
        
    #     mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), voxel_prim_path)
    #     mesh.CreatePointsAttr(Gf_Vec3f_list_of_vertices_tensor(vertices_from_centers(self.__voxel_centers[i,j,k])))
    #     mesh.CreateFaceVertexCountsAttr(12*[3]) # 12 tris per cube, 3 vertices each
    #     mesh.CreateFaceVertexIndicesAttr([3,4,2,  3,5,4,  3,7,5,  3,0,7,  3,1,0,  3,2,1,
    #                                       7,6,4,  7,4,5,  7,0,1,  7,1,6,  1,4,6,  1,2,4,])     
                                          
    #     return voxel_prim_path

def apply_color_to_prim(color: tuple) -> Callable[[str],None]:
    """Apply an RGB color to a prim. 

    Args:
        color (tuple): The RGB color to apply as a tuple of three floats.
    
    Lambda Args:
        prim_path (str): The path to the prim."""
    def __apply_color_to_prim_helper(prim_path : str) -> Tuple[float,float,float] :
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

    