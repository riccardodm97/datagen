import blenderproc as bproc
import os 
import pickle
import argparse
from pathlib import Path
import yaml
from typing import Tuple 
import shutil

import math
import numpy as np
from dotmap import DotMap
from transforms3d import affines, euler
import bpy



BASE_PATH = Path('~/dev').expanduser()      #TOCHANGE cosi è specifico per questo pc 
SCENE_PATH =  BASE_PATH/'data'/'scenes'
BLENDER_PATH = BASE_PATH/'data'/'blender_tmp'
DATASET_FOLDER = BASE_PATH/'data'/'datasets'



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
            -scanbox_dimensions.z/2,                    #CHANGED 0
            scanbox_dimensions.x / 2,
            scanbox_dimensions.y / 2,
            scanbox_dimensions.z/2,                     #CHANGED dimensions.z
        ]
    )
    return np.round(bbox, 4)


def points_on_dome(r, num_points):

    indices = np.arange(0, num_points, dtype=float) + 0.5

    theta = np.arccos(1 - 2*indices/num_points)
    phi = np.pi * (1 + 5**0.5) * indices

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    xyz = np.stack((x,y,z),axis=-1) 
    xyz = xyz[xyz[:,-1] > 0]  #take only points above poi 

    return xyz

def n_points_on_circle_evenly_spaced(r, center, num_points):

    theta = np.linspace(0,2*np.pi,num_points,endpoint=False); 
    x = r * np.cos(theta)+center[0]
    y = r * np.sin(theta)+center[1]
    z = np.broadcast_to(center[2],num_points)

    xyz = np.stack((x,y,z),axis=-1)
    return xyz

def points_circle(r, center, num_points):
    return [
            [center[0] + math.cos(2 * math.pi / num_points * x) * r, center[1] + math.sin(2 * math.pi / num_points * x) * r, center[2]]
                for x in range(0, num_points + 1)
            ]



#generate random camera poses on a dome around the scene 
def poses_on_dome(objs: list, num_poses: int, radius: float):
    '''
    generate random camera poses on a dome around the scene 
    '''  
    #point of intereset in the scene (the camera always face the poi)
    poi = bproc.object.compute_poi(objs)

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

#generate random camera poses on a dome around the scene with light in a fixed positon 
def camera_on_dome_uniformly(objs: list, num_poses: int, camera_radius: float, light_pos : list):    
    '''
    Generate random camera poses on a dome around the scene with the light in a fixed position
    '''  
    #point of intereset in the scene (the camera always face the poi)
    poi = bproc.object.compute_poi(objs)

    xyz_c = points_on_dome(camera_radius,num_poses*2)  #double it because we only take the z positive (above poi )
    xyz_c = xyz_c + poi 
    np.random.shuffle(xyz_c)

    light_tvec = np.array(light_pos)  
    light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - light_tvec)         #DEBUG 
    light2world  = bproc.math.build_transformation_mat(light_tvec, light_rotation_matrix)

    camera_poses = []
    light_poses = []
    for position in xyz_c : 
        cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        cam2world  = bproc.math.build_transformation_mat(position, cam_rotation_matrix)
        camera_poses.append(cam2world)

        light_poses.append(light2world)

    return camera_poses, light_poses

def camera_circle(objs: list, num_poses : int, camera_radius: float, circle_zenit: int, light_pos : list) -> Tuple:
    '''
    The light is on a fixed position in the scene determined by light_pos
    The camera goes on a circle around the poi at a given height (zenit) of a dome determined by camera_radius
    '''

    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    light_tvec = np.array(light_pos)  
    light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - light_tvec)        
    light2world  = bproc.math.build_transformation_mat(light_tvec, light_rotation_matrix)

    theta = np.radians(circle_zenit)
    circle_center = poi.copy()
    circle_center[2]+= camera_radius * np.cos(theta) 
    circle_radius = (camera_radius * np.sin(theta)) 
    xyz_c = n_points_on_circle_evenly_spaced(circle_radius,circle_center,num_poses)

    camera_poses = []
    light_poses = []
    for position in xyz_c:
        light_poses.append(light2world)

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        cam2world = bproc.math.build_transformation_mat(position, rotation_matrix)
        camera_poses.append(cam2world)

    return camera_poses, light_poses


#generate random camera poses on a dome, and light poses translated by t from camera
def camera_dome_translated_light(objs: list, num_poses : int, t_vector : list, dome_radius: float) -> Tuple:
    '''
    generate random camera poses on a dome, and light poses translated by t_vec from the camera
    '''

    t_vec = np.array(t_vector)   #convert the t_vector in numpy array  

    #generate translation matrix from translation vector
    t_matrix = np.eye(4)                                        
    t_matrix[:3,3] = t_vec

    #determine point of interest in the scene 
    poi = bproc.object.compute_poi(objs)

    camera_poses = []
    light_poses = []

    for _ in range(num_poses):
        cam_location = bproc.sampler.part_sphere(
            center=poi,
            radius=dome_radius,
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


#generate random camera poses on a dome, and n light poses translated by t_vec from camera
def camera_uniform_dome_n_translated_lights(objs: list, num_poses : int, t_vecs : list, dome_radius: float) -> Tuple:
    '''
    generate random camera poses on a dome, while light poses are translated by a list of t_vec from the camera
    '''

    t_vecs = np.array(t_vecs)   #convert the t_vector in numpy array  
    t_matrices = np.zeros((t_vecs.shape[0],4,4))

    #generate translation matrices from translation vector list
    for i, t_vec in enumerate(t_vecs) :
        t_matrices[i] = np.eye(4)                                        
        t_matrices[i,:3,3] = t_vec

    #determine point of interest in the scene 
    poi = bproc.object.compute_poi(objs)

    xyz_c = points_on_dome(dome_radius,(num_poses//len(t_vecs))*2)  #divide by number of light poses, double it because we only take the z positive (above poi )
    xyz_c = xyz_c + poi 
    np.random.shuffle(xyz_c)   #to avoid having the poses ordered from top to bottom 

    camera_poses = []
    light_poses = []
    for cam_location in xyz_c :
        cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
        cam2world  = bproc.math.build_transformation_mat(cam_location, cam_rotation_matrix)

        for t_matrix in t_matrices:
            camera_poses.append(cam2world)

            c2l = np.matmul(cam2world,t_matrix)
            light_location = c2l[:3,3]
            light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
            light2world_matrix = bproc.math.build_transformation_mat(light_location, light_rotation)
            light_poses.append(light2world_matrix)

    return camera_poses, light_poses


def translated_light_circle_fixed_camera(objs: list, num_poses : int, t_vector : list, dome_radius: float, dome_zenit: int) -> Tuple:
    '''
    generate poses for the light at translated positions from points on a circle at some height on a hemisphere around the poi with a given radius 
    generate poses for the camera at a fixed position on a dome
    theta is the zenit in degree.
    both camera and lights are on the same height from the poi 
    '''

    assert t_vector is not None, ' this method takes a translation for the light wrt to the camera'
    
    t_vec = np.array(t_vector)   #convert the t_vector in numpy array 
  
    #generate translation matrix from translation vector
    t_matrix = np.eye(4)
    t_matrix[:3,3] = t_vec

    poi = bproc.object.compute_poi(objs)

    camera_poses = []
    light_poses = []

    theta = math.radians(dome_zenit)

    poi_ = poi.copy()
    poi_[2] += dome_radius * math.cos(theta)
    cam_location = bproc.sampler.disk(
        center=poi_,
        radius=dome_radius * math.sin(theta),
        sample_from="circle",
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
    cam2world  = bproc.math.build_transformation_mat(cam_location, rotation_matrix)

    # compute a light trajectory
    lights_positions = points_circle(dome_radius, poi_, num_poses)
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


def camera_circle_fixed_translated_light(objs: list, num_poses : int, t_vector : list, dome_radius: float, dome_zenit: int) -> Tuple :
    '''
    generate poses for the camera on a circle at some height on a hemisphere around the poi with a given radius 
    generate poses for light at a fixed and translated position 
    dome_zenit is the zenit in degree which determines the circle height on the dome hemisphere 
    both camera and lights are on the same height from the poi 
    '''

    assert t_vector is not None, ' this method takes a translation for the light wrt to the camera'

    t_vec = np.array(t_vector)   #convert the t_vector in numpy array 
  
    #generate translation matrix from translation vector
    t_matrix = np.eye(4)
    t_matrix[:3,3] = t_vec

    poi = bproc.object.compute_poi(objs)

    camera_poses = []
    light_poses = []

    theta = math.radians(dome_zenit)

    poi_ = poi.copy()
    poi_[2] += dome_radius * math.cos(theta)
    light_location = bproc.sampler.disk(
        center=poi_,
        radius= dome_radius * math.sin(theta),
        sample_from="circle",
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - light_location)
    light2world = bproc.math.build_transformation_mat(light_location, rotation_matrix)
    light2light_translated = np.matmul(light2world, t_matrix)
    light_location = light2light_translated[:3,3]
    light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
    light_translated2world = bproc.math.build_transformation_mat(light_location, light_rotation)

    # compute a light trajectory
    camera_positions = points_circle(dome_radius, poi_, num_poses)
    for position in camera_positions:
        light_poses.append(light_translated2world)

        cam_rotation = bproc.camera.rotation_from_forward_vec(poi - position)
        cam2world = bproc.math.build_transformation_mat(position, cam_rotation)
        camera_poses.append(cam2world)

    return camera_poses, light_poses


def camera_uniformly_on_dome_and_circle_light(objs: list, tot_poses : int, num_pos_camera : int, num_pos_light : int, dome_radius: float, circle_zenit : int, circle_radius_delta : float):
    '''
    generate poses for the camera on a dome with a given radius while the light is at num_pos_light fixed positions on a cicle at a given zenit 
    with a radius which is bigger than the dome radius at a given zenit by a scalar addition of circle_radius_delta
    '''

    assert tot_poses == num_pos_camera * num_pos_light

    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    xyz_c = points_on_dome(dome_radius,num_pos_camera*2)  #double it because we only take the z positive (above poi )
    np.random.shuffle(xyz_c)   #to avoid having the poses ordered from top to bottom 
    xyz_c = xyz_c + poi 

    theta = np.radians(circle_zenit)
    circle_center = poi.copy()
    circle_center[2]+= dome_radius * np.cos(theta) #la dome è centrata in 0 ma poi tutti i punti vengono alzati di poi[2]
    circle_radius = dome_radius * np.sin(theta) + circle_radius_delta
    xyz_l = n_points_on_circle_evenly_spaced(circle_radius,circle_center,num_pos_light)


    camera_poses = []
    light_poses = []
    for c_id in range(xyz_c.shape[0]):
        cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - xyz_c[c_id])
        cam2world  = bproc.math.build_transformation_mat(xyz_c[c_id], cam_rotation_matrix)

        for l_id in range(xyz_l.shape[0]):
            camera_poses.append(cam2world)

            light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - xyz_l[l_id])
            light2world  = bproc.math.build_transformation_mat(xyz_l[l_id], light_rotation_matrix)
            light_poses.append(light2world)

    return camera_poses, light_poses


def camera_light_on_two_domes_uniformly(objs: list, num_poses : int, delta_domes : float, smaller_dome_radius: float):
    '''
    generate poses for the camera and the light on two domes where the bigger one (light dome) encloses the smaller one. Both are centered on 
    the point of interest. Camera and light poses are then randomly coupled to obtain a pair camera-light 
    Delta domes is the positive difference between the bigger dome radius and the smaller one
    '''

    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    xyz_c = points_on_dome(smaller_dome_radius,num_poses*2)  #double it because we only take the z positive (above poi )
    xyz_c = xyz_c + poi 
    np.random.shuffle(xyz_c)


    xyz_l = points_on_dome(smaller_dome_radius+delta_domes,num_poses*2)
    xyz_l = xyz_l + poi  

    c_idxs = np.arange(0,num_poses)
    l_idxs = c_idxs.copy()
    np.random.shuffle(l_idxs)

    camera_poses = []
    light_poses = []
    for c_id,l_id in zip(c_idxs,l_idxs):
        cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - xyz_c[c_id])
        cam2world  = bproc.math.build_transformation_mat(xyz_c[c_id], cam_rotation_matrix)
        camera_poses.append(cam2world)

        light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - xyz_l[l_id])
        light2world  = bproc.math.build_transformation_mat(xyz_l[l_id], light_rotation_matrix)
        light_poses.append(light2world)

    return camera_poses, light_poses

#TODO fix 
# def generate_poses_same_dome(objs: list, num_poses : int, dome_radius: float):
#     '''
#     generate poses for the camera and the light on the same dome centered on the point of interest. 
#     Camera and light poses are then randomly coupled to obtain a pair camera-light 
#     '''

#     #determine point of interest in the scene
#     poi = bproc.object.compute_poi(objs)

#     xyz_c = points_on_dome(dome_radius,num_poses*2)  #double it because we only take the z positive (above poi )
#     np.random.shuffle(xyz_c)
#     xyz_c = xyz_c + poi 

#     xyz_l = points_on_dome(dome_radius,num_poses*2)
#     np.random.shuffle(xyz_l)
#     xyz_l = xyz_l + poi  

#     camera_poses = []
#     light_poses = []
#     for c_id,l_id in zip(c_idxs,l_idxs):
#         cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - xyz_c[c_id])
#         cam2world  = bproc.math.build_transformation_mat(xyz_c[c_id], cam_rotation_matrix)
#         camera_poses.append(cam2world)

#         light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - xyz_l[l_id])
#         light2world  = bproc.math.build_transformation_mat(xyz_l[l_id], light_rotation_matrix)
#         light_poses.append(light2world)

#     return camera_poses, light_poses



def fixed_camera_on_dome_light_circle(objs: list, num_poses : int, camera_radius: float, circle_zenit: int, radius_delta : float) -> Tuple:
    '''
    The camera is on a fixed position on a dome of given radius while the light goes on a circle around the poi at a given height (zenit)
    radius_delta is used to define the offset between the camera radius (dome) and the light radius (circle)
    '''

    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    cam_location = bproc.sampler.part_sphere(
            center=poi,
            radius=camera_radius,
            part_sphere_dir_vector=[0, 0, 1],
            mode="SURFACE",
            dist_above_center=0.0,
        )
    
    cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
    cam2world  = bproc.math.build_transformation_mat(cam_location, cam_rotation_matrix)

    theta = np.radians(circle_zenit)
    circle_center = poi.copy()
    circle_center[2]+= (camera_radius + radius_delta) * np.cos(theta) 
    circle_radius = ((camera_radius + radius_delta) * np.sin(theta))
    xyz_l = points_circle(circle_radius,circle_center,num_poses)

    camera_poses = []
    light_poses = []
    for position in xyz_l:
        camera_poses.append(cam2world)

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        light2world = bproc.math.build_transformation_mat(position, rotation_matrix)
        light_poses.append(light2world)

    return camera_poses, light_poses


def fixed_light_on_dome_camera_circle(objs: list, num_poses : int, camera_radius: float, circle_zenit: int, radius_delta : float) -> Tuple:
    '''
    The light is on a fixed position on a dome of given radius while the camera goes on a circle around the poi at a given height (zenit)
    radius_delta is used to define the offset between the camera radius (circle) and the light radius (dome)
    '''

    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    light_location = bproc.sampler.part_sphere(
            center=poi,
            radius=camera_radius+radius_delta,
            part_sphere_dir_vector=[0, 0, 1],
            mode="SURFACE",
            dist_above_center=0.0,
        )
    
    light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - light_location)
    light2world  = bproc.math.build_transformation_mat(light_location, light_rotation_matrix)

    theta = np.radians(circle_zenit)
    circle_center = poi.copy()
    circle_center[2]+= camera_radius * np.cos(theta) 
    circle_radius = (camera_radius * np.sin(theta)) 
    xyz_c = points_circle(circle_radius,circle_center,num_poses)

    camera_poses = []
    light_poses = []
    for position in xyz_c:
        light_poses.append(light2world)

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        cam2world = bproc.math.build_transformation_mat(position, rotation_matrix)
        camera_poses.append(cam2world)

    return camera_poses, light_poses

def camera_from_above_light_circle(objs: list, num_poses : int, camera_radius: float, circle_zenit: int, radius_delta : float) -> Tuple:
    '''
    the camera is fixed at the top position of its dome (defined by camera radius) looking down on the scene. 
    the light goes on a circle around the poi at a given height (zenit) on the light dome (defined by radius_delta)
    '''
    
    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    theta = np.radians(0)
    cam_location = poi.copy()
    cam_location[2]+= camera_radius * np.cos(theta) 

    cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
    cam2world  = bproc.math.build_transformation_mat(cam_location, cam_rotation_matrix)

    theta2 = np.radians(circle_zenit)
    circle_center = poi.copy()
    circle_center[2]+= (camera_radius + radius_delta) * np.cos(theta2) 
    circle_radius = ((camera_radius + radius_delta) * np.sin(theta2))
    xyz_l = points_circle(circle_radius,circle_center,num_poses)

    camera_poses = []
    light_poses = []
    for position in xyz_l:
        camera_poses.append(cam2world)

        light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        light2world = bproc.math.build_transformation_mat(position, light_rotation_matrix)
        light_poses.append(light2world)

    return camera_poses, light_poses


def light_from_above_camera_circle(objs: list, num_poses : int, camera_radius: float, circle_zenit: int, radius_delta : float) -> Tuple:
    '''
    the light is fixed at the top position of its dome (defined by camera radius + radius_delta) illuminating the scene from above (noon style). 
    the camera goes on a circle around the poi at a given height (zenit) on the camera dome (defined by camera_radius)
    '''
    
    #determine point of interest in the scene
    poi = bproc.object.compute_poi(objs)

    theta = np.radians(0)
    light_location = poi.copy()
    light_location[2]+= (camera_radius + radius_delta) * np.cos(theta) 

    light_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - light_location)
    light2world  = bproc.math.build_transformation_mat(light_location, light_rotation_matrix)

    theta2 = np.radians(circle_zenit)
    circle_center = poi.copy()
    circle_center[2]+= camera_radius * np.cos(theta2) 
    circle_radius = camera_radius * np.sin(theta2)
    xyz_l = points_circle(circle_radius,circle_center,num_poses)

    camera_poses = []
    light_poses = []
    for position in xyz_l:
        light_poses.append(light2world)

        cam_rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        cam2world = bproc.math.build_transformation_mat(position, cam_rotation_matrix)
        camera_poses.append(cam2world)

    return camera_poses, light_poses




def main(config_file : str, dataset_id : str) :

    cfg_file = os.path.join(DATASET_FOLDER, config_file+'.yml')
    assert os.path.exists(cfg_file), 'config yaml file not found'

    with open(cfg_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotMap(cfg_dict,_dynamic=False)
    
    in_path = SCENE_PATH / (cfg.scene.input + '.blend')                  # path of the blender scene to load
    out_render_path = BLENDER_PATH / dataset_id / 'render'        # path where the rendered images will be stored as hdf5 files

    bproc.init()
    # Transparent Background
    bproc.renderer.set_output_format(enable_transparency=True)
    # Import just the objects and not the lights
    objs = bproc.loader.load_blend(str(in_path), data_blocks=["objects"], obj_types=["mesh"])
    bproc.camera.set_resolution(cfg.images.width, cfg.images.height)
    bproc.renderer.set_noise_threshold(16)
    #bproc.renderer.enable_depth_output(False)
    bproc.renderer.enable_distance_output(False,file_prefix='depth', output_key='depth')  # DEBUG enable depth (ray distance) output


    #load function from yaml
    kwargs_func = cfg.gen_function.toDict()   
    func_name = kwargs_func.pop('f')
    gen_func = globals()[func_name]
    camera_poses, light_poses = gen_func(objs,cfg.images.num,**kwargs_func)

    #light 
    light = bproc.types.Light(type=cfg.light.type, name = 'light')
    light.set_energy(cfg.light.energy)
    light.blender_obj.data.shadow_soft_size = cfg.light.radius

    for i in range(cfg.images.num):
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
            "image_size": [cfg.images.width, cfg.images.height],
        },
    }
    camera_metadata_path = BLENDER_PATH / dataset_id / 'camera_metadata.pickle' 
    with open(camera_metadata_path, 'wb') as m:
        pickle.dump(camera_metadata, m)

    bbox = get_scene_bbox(cfg.scene.delimiter_obj)
    bbox_path = BLENDER_PATH / dataset_id / 'bbox.npy'
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
    
    c_poses_path = BLENDER_PATH / dataset_id / 'c_poses.npy'
    l_poses_path = BLENDER_PATH / dataset_id / 'l_poses.npy'
    with open(c_poses_path,'wb') as c, open(l_poses_path,'wb') as l:
        np.save(c, c_poses)
        np.save(l, l_poses)
    
    metadata_path = BLENDER_PATH / dataset_id / 'metadata.yml'
    try : 
        shutil.copy(cfg_file, metadata_path)                # copy yaml cfg file to blender directory 
    except Exception as e: 
        print(str(e))
       


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_file', type=str, help='yml config file', required=True)
    parser.add_argument('--id', dest='dataset_id', type=str, help='id for the dataset', required=True)
    
    args = parser.parse_args()
    
    main(args.config_file,args.dataset_id)

    #main('gen_config','pepper_light')
  
