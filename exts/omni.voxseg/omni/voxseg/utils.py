import torch
from _VoxelGrid import VoxelGrid
import json
import rospy

############### ROS ##################

def convert_dict_to_dictionary_array(dictionary):

    # Convert the Python dictionary to serialized JSON strings
    dictionary_jsons = [json.dumps({key: value}) for key, value in dictionary.items()]

    # Convert the serialized JSON strings to ROS String messages
    dictionary_json_msgs = [str(json_str) for json_str in dictionary_jsons]


    return dictionary_json_msgs

def convert_dictionary_array_to_dict(dictionary_jsons):
    # Create an empty dictionary to store the result
    result_dict = {}

    # Deserialize the JSON strings and populate the result dictionary
    for json_str in dictionary_jsons:
        try:
            data = json.loads(json_str)
            result_dict.update(data)
        except ValueError as e:
            rospy.logerr("Error while deserializing JSON string: %s", str(e))

    return result_dict

def voxels_from_msg(msg: VoxelGrid):
    voxels: torch.Tensor = torch.as_tensor(msg.data).float()
    voxels -= 1 # the backend adds 1 to the voxel classes, in order to encode the array in bytes
    voxel_grid_shape = (msg.size_x, msg.size_y, msg.size_z)
    voxels = voxels.view(*voxel_grid_shape)

    resolutions = torch.as_tensor([msg.resolutions.x, msg.resolutions.y, msg.resolutions.z])
    grid_dim = torch.as_tensor(tuple(voxels.size()))
    world_dim = grid_dim / resolutions
    return voxels, world_dim
