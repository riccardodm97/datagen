import blenderproc as bproc
import os 
import pickle
import argparse
from pathlib import Path
import yaml
from typing import Tuple 

import math
import numpy as np
from dotmap import DotMap
from transforms3d import affines, euler
import bpy



BASE_PATH = Path('~/dev').expanduser()      #TOCHANGE cosi è specifico per questo pc 
SCENE_PATH =  BASE_PATH/'data'/'scenes'
BLENDER_PATH = BASE_PATH/'data'/'blender'
CONFIG_FOLDER = BASE_PATH/'data'/'dataset'/'config'



# get thte bbox of the scene enclosed in a delimiter object
def get_scene_bbox(delimiter_obj: str) -> np.ndarray:
    """
    :param obj_bbox: _description_
    :type obj_bbox: str
    :return: _description_
    :rtype: np.array
    """
    scanbox_dimensions = bpy.data.objects[delimiter_obj].dimensions
    bbox = np.array(
        [
            -scanbox_dimensions.x / 2,
            -scanbox_dimensions.y / 2,
            0,                                   #TOCHANGE ? dimensions.z/2 ??  
            scanbox_dimensions.x / 2,
            scanbox_dimensions.y / 2,
            scanbox_dimensions.z,
        ]
    )
    return np.round(bbox, 4)

#generate random camera poses on a dome around the scene 
def generate_poses_on_dome_camera(objs: list, radius: float, num_poses: int, poi = None):
    '''
    generate random camera poses on a dome around the scene 
    '''  
    #point of intereset in the scene (the camera always face the poi)
    poi = poi or bproc.object.compute_poi(objs)

    poses = []
    for _ in range(num_poses):
        location = bproc.sampler.part_sphere(
            center=poi,
            radius=radius,
            part_sphere_dir_vector=[0, 0, 1],
            mode="SURFACE",
            dist_above_center=0.0,
        )
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        poses.append(cam2world_matrix)
        

    return poses


#generate random camera poses on a dome, and light poses translated by t from camera
def generate_poses_on_dome_camera_and_translated_light(objs: list, t_vec: np.array, radius: float, num_poses: int) -> Tuple:
    '''
    generate random camera poses on a dome, and light poses translated by t_vec from the camera
    ''' 

    #generate translation matrix from translation vector
    t_matrix = np.eye(4)                                        #TODO fetch the translation vector from the cfg file 
    t_matrix[:3,3] = t_vec

    #determine point of interest in the scene 
    poi = bproc.object.compute_poi(objs)

    camera_poses = []
    light_poses = []

    for _ in range(num_poses):
        cam_location = bproc.sampler.part_sphere(
            center=poi,
            radius=radius,
            part_sphere_dir_vector=[0, 0, 1],
            mode="SURFACE",
            dist_above_center=0.0,
        )
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
        cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
        camera_poses.append(cam2world_matrix)

        c2l = cam2world_matrix @ t_matrix
        light_location = c2l[:3,3]
        light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
        light2world_matrix = bproc.math.build_transformation_mat(light_location, light_rotation)
        light_poses.append(light2world_matrix)

    return camera_poses, light_poses


# generate poses for the light on a circle at some height on a hemisphere around the poi with a given radius
def generate_poses_on_circle_light_and_fixed_camera(objs: list, t_light_cfg, radius: float, theta : int, num_poses: int) -> Tuple:
    '''
    generate poses for the light on a circle at some height on a hemisphere around the poi with a given radius and fixed camera position. 
    theta is the zenit in degree
    '''

    assert t_light_cfg.is_active == True, ' this method takes a translation for the light wrt to the camera'

    def points_circle(r, center, num_points):
        return [
            [center[0] + math.cos(2 * math.pi / num_points * x) * r, center[1] + math.sin(2 * math.pi / num_points * x) * r, center[2]]
                for x in range(0, num_points + 1)
            ]
    
    t_vec = np.array(t_light_cfg.vector)   #fetch the tranlsation vector from the cfg file 
  
    #generate translation matrix from translation vector
    t_matrix = np.eye(4)
    t_matrix[:3,3] = t_vec

    poi = bproc.object.compute_poi(objs)

    camera_poses = []
    light_poses = []

    theta = math.radians(theta)

    poi_ = poi.copy()
    poi_[2] += radius * math.cos(theta)
    cam_location = bproc.sampler.disk(
        center=poi_,
        radius=radius * math.sin(theta),
        sample_from="circle",
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
    cam2world  = bproc.math.build_transformation_mat(cam_location, rotation_matrix)

    # compute a light trajectory
    lights_positions = points_circle(radius, poi_, num_poses)
    for position in lights_positions:
        camera_poses.append(cam2world)

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        light2world = bproc.math.build_transformation_mat(position, rotation_matrix)
        light2light_translated = np.matmul(light2world, t_matrix)
        light_location = light2light_translated[:3,3]
        light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
        light_translated2world = bproc.math.build_transformation_mat(light_location, light_rotation)
        light_poses.append(light_translated2world)

    return camera_poses, light_poses


def generate_poses_on_circle_camera_and_fixed_light(objs: list, t_light_cfg, radius: float, theta: int, num_poses: int) :
    '''
    generate poses for the camera on a circle at some height on a hemisphere around the poi with a given radius and fixed light position
    theta is the zenit in degree
    '''

    assert t_light_cfg.is_active == True, ' this method takes a translation for the light wrt to the camera'

    def points_circle(r, center, num_points):
        return [
            [center[0] + math.cos(2 * math.pi / num_points * x) * r, center[1] + math.sin(2 * math.pi / num_points * x) * r, center[2]]
                for x in range(0, num_points + 1)
            ]

    t_vec = np.array(t_light_cfg.vector)   #fetch the tranlsation vector from the cfg file 
  
    #generate translation matrix from translation vector
    t_matrix = np.eye(4)
    t_matrix[:3,3] = t_vec

    poi = bproc.object.compute_poi(objs)

    camera_poses = []
    light_poses = []

    theta = math.radians(theta)

    poi_ = poi.copy()
    poi_[2] += radius * math.cos(theta)
    light_location = bproc.sampler.disk(
        center=poi_,
        radius=radius * math.sin(theta),
        sample_from="circle",
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - light_location)
    light2world = bproc.math.build_transformation_mat(light_location, rotation_matrix)
    light2light_translated = np.matmul(light2world, t_matrix)
    light_location = light2light_translated[:3,3]
    light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
    light_translated2world = bproc.math.build_transformation_mat(light_location, light_rotation)

    # compute a light trajectory
    camera_positions = points_circle(radius, poi_, num_poses)
    for position in camera_positions:
        light_poses.append(light_translated2world)

        cam_rotation = bproc.camera.rotation_from_forward_vec(poi - position)
        cam2world = bproc.math.build_transformation_mat(position, cam_rotation)
        camera_poses.append(cam2world)

    return camera_poses, light_poses



def main(config_file : str) :

    cfg_file = os.path.join(CONFIG_FOLDER,config_file+'.yml')
    assert os.path.exists(cfg_file), 'config yaml file not found'

    with open(cfg_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotMap(cfg_dict,_dynamic=False)

    in_path = SCENE_PATH / (cfg.input_scene + '.blend')                  # path of the blender scene to load
    out_render_path = BLENDER_PATH / cfg.output_folder / 'render'        # path where the rendered images will be stored as h2f5 files

    bproc.init()
    # Transparent Background
    bproc.renderer.set_output_format(enable_transparency=True)
    # Import just the objects and not the lights
    objs = bproc.loader.load_blend(str(in_path), data_blocks=["objects"], obj_types=["mesh"])
    bproc.camera.set_resolution(cfg.image.width, cfg.image.height)
    bproc.renderer.set_noise_threshold(16)
    bproc.renderer.enable_depth_output(False)

    
    camera_poses, light_poses = generate_poses_on_circle_camera_and_fixed_light(objs,cfg.translate_light,cfg.dome.radius,cfg.dome.zenit,cfg.num_images)

    #light 
    light = bproc.types.Light(type='POINT', name = 'light')
    light.set_energy(1000)

    for i in range(cfg.num_images):
        frame = bpy.context.scene.frame_end
        bproc.camera.add_camera_pose(camera_poses[i])
        t, r, _, _ = affines.decompose(light_poses[i])
        r_euler = euler.mat2euler(r)
        light.set_location(t,frame)
        light.set_rotation_euler(r_euler,frame)
    
    data = bproc.renderer.render()


    #save everything to temp blender folder to be later converted into underfolder 

    bproc.writer.write_hdf5(out_render_path, data)

    camera_metadata = {
        "camera_pose": {
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 0],
        },
        "intrinsics": {
            "camera_matrix": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
            "dist_coeffs": np.zeros((5, 1)).tolist(),
            "image_size": [cfg.image.width, cfg.image.height],
        },
    }
    metadata_path = BLENDER_PATH / cfg.output_folder / 'camera_metadata.pickle' 
    with open(metadata_path, 'wb') as m:
        pickle.dump(camera_metadata, m)

    bbox = get_scene_bbox(cfg.delimiter_obj)
    bbox_path = BLENDER_PATH / cfg.output_folder / 'bbox.npy'
    with open(bbox_path, 'wb') as f:
        np.save(f, bbox)

    c_poses, l_poses = [], []
    for c_pose, l_pose in zip(camera_poses,light_poses): #DEBUG
        camera_pose = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
            c_pose, ["X", "-Y", "-Z"]
        )
        light_pose = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
            l_pose, ["X", "-Y", "-Z"]
        )
        c_poses.append(camera_pose)
        l_poses.append(light_pose)
    
    c_poses_path = BLENDER_PATH / cfg.output_folder / 'c_poses.npy'
    l_poses_path = BLENDER_PATH / cfg.output_folder / 'l_poses.npy'
    with open(c_poses_path,'wb') as c, open(l_poses_path,'wb') as l:
        np.save(c, c_poses)
        np.save(l, l_poses)
       


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', dest='config_file', type=str, help='yml config file', required=True)
    
    # args = parser.parse_args()
    
    # main(args.config_file)

    main('camera360')
  
