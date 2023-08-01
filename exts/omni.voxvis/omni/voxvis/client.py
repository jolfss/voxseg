import rospy
from std_msgs.msg import String, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image

CATKIN_PATH = '/home/pcgta/Documents/eth/voxseg' # TODO: Maybe have this exported and get from environment

import sys
sys.path.append(f'{CATKIN_PATH}/devel/lib/python3/dist-packages/voxseg/msg')

from _DepthImageInfo import DepthImageInfo
from _Classes import Classes
from _VoxelGrid import VoxelGrid
from _WorldInfo import WorldInfo

sys.path.append(f'{CATKIN_PATH}/devel/lib/python3/dist-packages/voxseg/srv')

from _VoxelComputation import VoxelComputation

#from voxseg.srv import VoxelComputation

import numpy as np
import torch
from typing import List, Dict, Union

from .config import CLASS_TOPIC, WORLD_DIM_TOPIC, VOXEL_REQUEST_SERVICE
from .utils import voxels_from_msg, convert_dict_to_dictionary_array

class VoxSegClient:
    def __init__(self):
        """
        initialize the frontend node
        """

        #rospy.init_node(CLIENT_NODE, anonymous=True)

        # Important: initialize the pubs before starting to publish
        self.class_pub = rospy.Publisher(CLASS_TOPIC, Classes, queue_size=10)
        self.dim_pub = rospy.Publisher(WORLD_DIM_TOPIC, WorldInfo, queue_size=10)
        
    def publish_world_info(self, world_dim, grid_dim):
        """
        Inputs:
            world_dim: arraylike, shape (3,), represents the xyz dimensions of the ground truth world
            grid_dim: arraylike containing ints, shape (3,), represents the xyz dimensions of the voxel grid
        """
        msg = WorldInfo(world_dim=list(world_dim), grid_dim=list(grid_dim))
        self.dim_pub.publish(msg)

    def publish_depth_image(self, image, depth_map, extrinsics):
        """
        Should be called alongside image_callback in the simulation
        Inputs:
            image: a numpy array containing rgb image data, shape (h,w,c)

            depth_map: a numpy array containing depth data, size (h,w)

            extrinsics: a numpy array containing camera extrinsics, size (4,4)

        """
        timestamp = rospy.Time.now()
        image_msg = self._get_image_msg(image, timestamp)
        depth_msg = self._get_depth_msg(depth_map, timestamp)
        extrinsics_msg = self._get_extrinsics_msg(extrinsics, timestamp)

        full_msg = DepthImageInfo()
        full_msg.rgb_image = image_msg
        full_msg.depth_image = depth_msg
        full_msg.cam_extrinsics = extrinsics_msg

        
        #pub = rospy.Publisher(IMAGE_TOPIC, DepthImageInfo, queue_size=10)
        self.image_pub.publish(full_msg)

    def publish_class_names(self, names: Union[List[str], None]=['other'], 
                            groups:Union[Dict[str, List[str]], None]=None,
                            prompts: Union[Dict[str, List[str]], None]=None, 
                            use_prompts=False):
        """
        Should be called whenever the user enters class names in the extension window
        
        Either names or prompts must be specified.
        
        Inputs:
            names: list of class identifiers. If None, defaults to the identifiers in prompts

            prompts: dictionary of class identifier to corresponding prompts
            
            groups: dictionary of group identifier to corresponding class identifiers. Default behavior is that each class identifier gets its own group.
            
            use_prompts: True to use the user-defined prompts, False to use automatically generated prompts with names
        """
        class_msg = Classes()

        class_msg.use_prompts = use_prompts
        if use_prompts:
            if prompts == None:
                raise Exception('use_prompts set to True, but no prompts were specified')
            class_msg.prompts = convert_dict_to_dictionary_array(prompts)
            class_msg.classes = list(prompts.keys())
        else:
            class_msg.classes = names

        if not groups:
            class_msg.groups = convert_dict_to_dictionary_array({name: [name] for name in class_msg.classes})
        else:
            class_msg.groups = convert_dict_to_dictionary_array(groups)


        self.class_pub.publish(class_msg)
    def request_voxel_computation(self, min_pts_in_voxel=0):
        """
        Publishes the VoxelGrid message returned by the request to VOXEL_TOPIC

        Inputs:
            min_pts_in_voxel: the minimum number of points to consider a voxel valid for inference
        
        Returns:
            torch.tensor representing the voxels and torch.tensor representing the world dim
        """
        print('Requesting Voxels')
        rospy.wait_for_service(VOXEL_REQUEST_SERVICE)
        try:
            compute_data_service = rospy.ServiceProxy(VOXEL_REQUEST_SERVICE, VoxelComputation)
            voxel_response = compute_data_service(min_pts_in_voxel)
            voxels = torch.as_tensor(voxel_response.data)
            print('VOXELS RECEIVED')
            return voxels
        except rospy.ServiceException as e:
            print('Service Call Failed')
            rospy.logerr("Service call failed: %s", e)
            return None

    def _get_extrinsics_msg(self, extrinsics, timestamp):
        return np.reshape(extrinsics, (16,)).tolist()

    def _get_image_msg(self, image, timestamp) -> Image:
        """
        Inputs:
            image: a numpy array containing rgb image data, shape (h,w,c)

            depth_map: a numpy array containing depth data, size (h,w)

            extrinsics: a numpy array containing camera extrinsics, size (4,4)

        """
        h,w,c = image.shape
        img_msg = Image()
        img_msg.width = w
        img_msg.height = h
        img_msg.encoding = "rgb8"  # Set the encoding to match your image format
        img_msg.data = image.tobytes()
        img_msg.header.stamp = timestamp
        img_msg.header.frame_id = 'img_frame'

        return img_msg


    def _get_depth_msg(self, depth_map, timestamp) -> Image:
        """
        depth_map: a numpy array containing depth data, size (h,w)
        """
        h,w = depth_map.shape
        depth_msg = Image()
        depth_msg.height = h
        depth_msg.width = w
        depth_msg.encoding = '32FC1'  # Assuming single-channel depth map
        depth_msg.step = w * 4  # Size of each row in bytes
        depth_msg.data = depth_map.astype(np.float32).tobytes()
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = 'depth_frame'

        return depth_msg




if __name__=='__main__':
    try:
        frontend = VoxSegClient()
    except rospy.ROSInterruptException:
        pass