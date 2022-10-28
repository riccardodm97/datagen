
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pipelime.sequences.readers.filesystem import UnderfolderReader
from sklearn.metrics import pairwise_distances

from utils import (makeLookAt, n_points_on_circle, n_points_on_vertical_circle,
                   nearest_intersection, points_circle, points_on_dome)


def samples_uniformly_hemisphere():

    u = np.random.uniform(low=0.0,high=1.0,size=100)
    v = np.random.uniform(low=0.0,high=1.0,size=100)

    radius_c = 0.8
    poi = np.array([0.8,0.2,0])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x_c,y_c,z_c = b(radius_c,u,v)
  
    ax.scatter(x_c,y_c,z_c,c='blue')
    ax.scatter(*poi, c='green')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def poses_uniformly_on_dome_and_circle_light():


    radius_c = 0.65
    poi = np.array([0.8,0.2,0.2])


    xyz_c = points_on_dome(radius_c,200)
    xyz_c = xyz_c + poi                       #DEBUG

    theta = np.radians(70)
    circle1_center = poi.copy()
    circle1_center[2]+= radius_c * np.cos(theta)                #DEBUG = oppure += ? la dome è centrata in 0 ma poi tutti i punti vengono alzati di poi[2]
    circle1_radius = radius_c * np.sin(theta) + 0.4
    xyz_l1 = n_points_on_circle(circle1_radius,circle1_center,6)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_c[:,0],xyz_c[:,1],xyz_c[:,2],c='blue')
    ax.scatter(xyz_l1[:,0],xyz_l1[:,1],xyz_l1[:,2],c='red')
    ax.scatter(*poi, c='black')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()



def camera_light_on_two_domes_uniformly():


    radius_c = 0.65
    radius_l = radius_c + 0.4
    poi = np.array([0.1,0.2,0.1])


    xyz_c = points_on_dome(radius_c,200)
    xyz_c = xyz_c + poi                       #DEBUG

    xyz_l = points_on_dome(radius_l,200)
    xyz_l = xyz_l + poi  


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_c[:,0],xyz_c[:,1],xyz_c[:,2],c='blue')
    ax.scatter(xyz_l[:,0],xyz_l[:,1],xyz_l[:,2],c='red')
    ax.scatter(*poi, c='green')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


# camera_light_on_two_domes_uniformly()


def fixed_camera_on_dome_light_circle():


    radius_c = 0.65
    radius_l = radius_c + 0.4
    poi = np.array([0.1,0.2,0.1])


    xyz_c = points_on_dome(radius_c,200)
    xyz_c = xyz_c + poi                  

    xyz_d = points_on_dome(radius_l,400)
    xyz_d = xyz_d + poi

    idx = np.random.randint(0, xyz_c.shape[0])
    camera_pos = xyz_c[idx,:]

    theta = np.radians(60)
    circle_center = poi.copy()
    circle_center[2]+= (radius_c + 0.4) * np.cos(theta) 
    circle_radius = ((radius_c + 0.4) * np.sin(theta)) 
    xyz_l = points_circle(circle_radius,circle_center,100)

    theta2 = np.radians(60)
    circle_center2 = poi.copy()
    circle_center2[2]+= (radius_c) * np.cos(theta2) + 0.4
    circle_radius2 = ((radius_c) * np.sin(theta2)) + 0.4
    xyz_l2 = points_circle(circle_radius2,circle_center2,100)


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_c[:,0],xyz_c[:,1],xyz_c[:,2],c='black')
    ax.scatter(xyz_l[:,0],xyz_l[:,1],xyz_l[:,2],c='blue')
    ax.scatter(xyz_d[:,0],xyz_d[:,1],xyz_d[:,2],c='purple')
    ax.scatter(xyz_l2[:,0],xyz_l2[:,1],xyz_l2[:,2],c='yellow')
    ax.scatter(*poi, c='green')
    ax.scatter(camera_pos[0],camera_pos[1],camera_pos[2],c='red')
    ax.plot([camera_pos[0],poi[0]], [camera_pos[1],poi[1]],[camera_pos[2],poi[2]], linestyle="--")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def light_from_above_camera_circle():


    radius_c = 0.65
    radius_l = radius_c + 0.4
    poi = np.array([0.1,0.2,0.1])

    theta = np.radians(0)
    camera_pos = poi.copy()
    camera_pos[2]+= radius_c * np.cos(theta) 

    theta2 = np.radians(70)
    circle_center = poi.copy()
    circle_center[2]+= radius_l * np.cos(theta2) 
    circle_radius = radius_l * np.sin(theta2)
    xyz_l = points_circle(circle_radius,circle_center,200)


    xyz_d1 = points_on_dome(radius_c,200)
    xyz_d1 = xyz_d1 + poi                  

    xyz_d2 = points_on_dome(radius_l,400)
    xyz_d2 = xyz_d2 + poi


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_d1[:,0],xyz_d1[:,1],xyz_d1[:,2],c='blue')
    ax.scatter(xyz_d2[:,0],xyz_d2[:,1],xyz_d2[:,2],c='purple')
    ax.scatter(camera_pos[0],camera_pos[1],camera_pos[2],c='red')
    ax.scatter(xyz_l[:,0],xyz_l[:,1],xyz_l[:,2],c='blue')
    ax.scatter(*poi, c='green')
    ax.plot([camera_pos[0],poi[0]], [camera_pos[1],poi[1]],[camera_pos[2],poi[2]],  linestyle="--")
  
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def camera_dome_light_ring():

    radius_c = 1.0
    radius_l = 0.2
    poi = np.array([0.1,0.2,0.1])

    xyz_c = points_on_dome(radius_c,100)
    xyz_c = xyz_c + poi                  

    one_xyz_l = n_points_on_vertical_circle(radius_l,xyz_c[30],30)
    
    xyz_del = np.delete(xyz_c,30,0)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(one_xyz_l[:,0],one_xyz_l[:,1],one_xyz_l[:,2],c='blue')
    ax.scatter(xyz_del[:,0],xyz_del[:,1],xyz_del[:,2],c='purple')
    ax.scatter(xyz_c[30,0],xyz_c[30,1],xyz_c[30,2],c='red')
    ax.scatter(*poi, c='green')

  
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
    
def plot_poses(
        poses: Sequence[np.ndarray], scale: Optional[float] = None, labels: bool = False
    ) -> None:
        poses_arr = np.stack(poses)
        tvecs = poses_arr[:, :3, 3]

        t_vectors = poses[:, :3, 3]
        z_vectors = poses[:, :3, 2]

        if scale is None:
            dists = pairwise_distances(tvecs, metric="euclidean")
            dists += np.eye(dists.shape[0]) * dists.max()
            scale = dists.min(0).mean()

        poses_arr[:, :3, :3] *= scale
        xyz = [poses_arr[:, :3, x] for x in range(3)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for vecs, color in zip(xyz, ["r", "g", "b"]):
            vectors = np.concatenate([tvecs, vecs], axis=1)
            X, Y, Z, U, V, W = zip(*vectors)
            ax.quiver(X, Y, Z, U, V, W, color=color)

        if labels:
            for i, (x, y, z) in enumerate(tvecs):
                ax.text(x, y, z, str(i))
        
        c = nearest_intersection(t_vectors,z_vectors)
        ax.scatter(*c, c='green')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def show_poses(
    input_folder: Path,
    pose_key: str,
    errors : np.ndarray = None,
    scale: Optional[float] = None,
    labels: bool = False
) -> None:

    matplotlib.use("TkAgg")

    uf = UnderfolderReader(input_folder)
    poses = []
    for sample in uf:
        pose: np.ndarray = sample[pose_key]
        if pose[3, 3] == 1:
            poses.append(pose)

    poses = np.array(poses)
    t_vectors = poses[:,:3,3] 
    distances = np.linalg.norm(t_vectors,axis=1)
    print(f'distance from origin -> min: {np.min(distances)}, max: {np.max(distances)}')

    n_poses = len(poses)
    n_unique_poses = len(np.unique(poses,axis=0))
    print(f'unique {pose_key}_key poses : {n_unique_poses}, which is {n_unique_poses/n_poses*100}% of all poses' )

    xs = [p[0,3] for p in poses]
    ys = [p[1,3] for p in poses]
    zs = [p[2,3] for p in poses]

    print(f'Tx -> min: {np.min(xs)} max: {np.max(xs)}')
    print(f'Ty -> min: {np.min(ys)} max: {np.max(ys)}')
    print(f'Tz -> min: {np.min(zs)} max: {np.max(zs)}')

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    if errors is None : 
        colors = np.ones(poses.shape[0])
    else : 
        colors = errors 
    cmhot = plt.get_cmap("hot")

    p = ax.scatter(poses[:,0,3],poses[:,1,3],poses[:,2,3],c=colors,cmap=cmhot)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if errors is not None : 
        cbar = fig.colorbar(p)
        cbar.set_label('error')

    plt.show()
    plot_poses(poses, scale, labels=labels)

#show_poses('/home/eyecan/dev/nerf_relight/real_relight/data/datasets/test/prova_marco/light360/uf','light')
    

def show_test_light(input_folder : Path, zenit, num_poses):

    uf = UnderfolderReader(input_folder)
    camera_poses = []
    light_poses = []
    for sample in uf:
        c_pose : np.ndarray = sample['pose']
        l_pose : np.ndarray = sample['light']
        if c_pose[3, 3]== 1 and l_pose[3,3]== 1:
            camera_poses.append(c_pose)
            light_poses.append(l_pose)

    camera_poses = np.array(camera_poses)
    light_poses = np.array(light_poses)


    light_t_vectors = light_poses[:,:3,3]
    light_z_vectors = light_poses[:,:3,2]

    light_dome_center = nearest_intersection(light_t_vectors,light_z_vectors)
    light_dome_center = light_dome_center.squeeze(1)
    light_dome_radius =  np.linalg.norm(light_t_vectors[0] - light_dome_center)


    camera_t_vectors = camera_poses[:,:3,3]
    camera_z_vectors = camera_poses[:,:3,2]

    camera_dome_center = nearest_intersection(camera_t_vectors,camera_z_vectors)
    camera_dome_center = camera_dome_center.squeeze(1)

    # determine light circle poses 
    theta = np.radians(zenit)
    circle_center = light_dome_center.copy()
    circle_center[2]+= light_dome_radius * np.cos(theta)            
    circle_radius = light_dome_radius * np.sin(theta)
    xyz_l = n_points_on_circle(circle_radius,circle_center,num_poses)

    test_light_poses = []

    for l in xyz_l : 

        light2world = makeLookAt(l,light_dome_center,[0,0,1])
        test_light_poses.append(light2world)
    
    test_light_poses = np.array(test_light_poses)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # light plot 
    light_x_line_endpoints = [light_dome_center[0], light_t_vectors[0][0]]
    light_y_line_endpoints = [light_dome_center[1], light_t_vectors[0][1]]
    light_z_line_endpoints = [light_dome_center[2], light_t_vectors[0][2]]
    ax.plot(light_x_line_endpoints, light_y_line_endpoints, light_z_line_endpoints, 'bo', linestyle="--")

    ax.scatter(test_light_poses[:,0,3],test_light_poses[:,1,3],test_light_poses[:,2,3],c='red')
    ax.scatter(*circle_center, c='black')
    ax.scatter(light_poses[:,0,3],light_poses[:,1,3],light_poses[:,2,3],c ='green')

    # camera plot 
    camera_x_line_endpoints = [camera_dome_center[0], camera_t_vectors[0][0]]
    camera_y_line_endpoints = [camera_dome_center[1], camera_t_vectors[0][1]]
    camera_z_line_endpoints = [camera_dome_center[2], camera_t_vectors[0][2]]
    ax.plot(camera_x_line_endpoints, camera_y_line_endpoints, camera_z_line_endpoints, 'bo', linestyle="--")

    ax.scatter(camera_poses[:,0,3],camera_poses[:,1,3],camera_poses[:,2,3],c ='orange')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0.1, 0.6])
    
    plot_poses(np.array(test_light_poses), scale=0.3)
    plt.show()

#show_test_light('/home/eyecan/dev/nerf_relight/real_relight/data/datasets/train/TM/prova_marco/uf3',38, 100)

def show_test_camera(input_folder : Path, zenit, num_poses):

    uf = UnderfolderReader(input_folder)
    camera_poses = []
    light_poses = []
    for sample in uf:
        c_pose : np.ndarray = sample['pose']
        l_pose : np.ndarray = sample['light']
        if c_pose[3, 3]== 1 and l_pose[3,3]== 1:
            camera_poses.append(c_pose)
            light_poses.append(l_pose)

    camera_poses = np.array(camera_poses)
    light_poses = np.array(light_poses)


    light_t_vectors = light_poses[:,:3,3]
    light_z_vectors = light_poses[:,:3,2]

    light_dome_center = nearest_intersection(light_t_vectors,light_z_vectors)
    light_dome_center = light_dome_center.squeeze(1)
    light_dome_radius =  np.linalg.norm(light_t_vectors[0] - light_dome_center)


    camera_t_vectors = camera_poses[:,:3,3]
    camera_z_vectors = camera_poses[:,:3,2]

    camera_dome_center = nearest_intersection(camera_t_vectors,camera_z_vectors)
    camera_dome_center = camera_dome_center.squeeze(1)
    camera_dome_radius =  np.linalg.norm(camera_t_vectors[0] - camera_dome_center)

    # determine camera circle poses 
    theta = np.radians(zenit)
    circle_center = camera_dome_center.copy()
    circle_center[2]+= camera_dome_radius * np.cos(theta)            
    circle_radius = camera_dome_radius * np.sin(theta)
    xyz_l = n_points_on_circle(circle_radius,circle_center,num_poses)

    test_camera_poses = []

    for l in xyz_l : 

        cam2world = makeLookAt(l,camera_dome_center,[0,0,1])
        test_camera_poses.append(cam2world)
    
    print(test_camera_poses.shape)

    x_min, x_max = np.min(camera_poses[:,0,3]), np.max(camera_poses[:,0,3])
    min_mask_x = test_camera_poses[:,0,3] >= x_min
    max_mask_x = test_camera_poses[:,0,3] <= x_max

    y_min, y_max = np.min(camera_poses[:,1,3]), np.max(camera_poses[:,1,3])
    min_mask_y = test_camera_poses[:,1,3] >= y_min
    max_mask_y = test_camera_poses[:,1,3] <= y_max

    mask = np.bitwise_and(np.bitwise_and(min_mask_x,max_mask_x),np.bitwise_and(min_mask_y,max_mask_y))
    test_camera_poses = test_camera_poses[mask]

    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # light plot 
    light_x_line_endpoints = [light_dome_center[0], light_t_vectors[0][0]]
    light_y_line_endpoints = [light_dome_center[1], light_t_vectors[0][1]]
    light_z_line_endpoints = [light_dome_center[2], light_t_vectors[0][2]]
    ax.plot(light_x_line_endpoints, light_y_line_endpoints, light_z_line_endpoints, 'bo', linestyle="--")

    # ax.scatter(test_light_poses[:,0,3],test_light_poses[:,1,3],test_light_poses[:,2,3],c='red')
    # ax.scatter(*circle_center, c='black')
    ax.scatter(light_poses[:,0,3],light_poses[:,1,3],light_poses[:,2,3],c ='green')

    # camera plot 
    camera_x_line_endpoints = [camera_dome_center[0], camera_t_vectors[0][0]]
    camera_y_line_endpoints = [camera_dome_center[1], camera_t_vectors[0][1]]
    camera_z_line_endpoints = [camera_dome_center[2], camera_t_vectors[0][2]]
    ax.plot(camera_x_line_endpoints, camera_y_line_endpoints, camera_z_line_endpoints, 'bo', linestyle="--")

    ax.scatter(test_camera_poses[:,0,3],test_camera_poses[:,1,3],test_camera_poses[:,2,3],c='red')
    ax.scatter(*circle_center, c='black')
    ax.scatter(camera_poses[:,0,3],camera_poses[:,1,3],camera_poses[:,2,3],c ='orange')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0.1, 0.6])
    
    plot_poses(np.array(test_camera_poses), scale=0.3)
    plt.show()


show_test_camera('/home/eyecan/dev/nerf_relight/real_relight/data/datasets/train/TM/prova_marco/uf3',68, 100)
