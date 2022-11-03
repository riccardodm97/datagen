import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer as t
from pipelime.sequences.operations import OperationResetIndices
from pipelime.sequences.readers.filesystem import UnderfolderReader
from pipelime.sequences.samples import PlainSample, SamplesSequence
from pipelime.sequences.writers.filesystem import (UnderfolderWriter,
                                                   UnderfolderWriterV2)
from typer import Typer

from utils.utils import makeLookAt, n_points_on_circle, nearest_intersection

FUNC = Typer(name="nome", pretty_exceptions_enable=False)


POSE_KEY = "pose"
CAMERA_KEY = "camera"
IMAGE_KEY = "image"
LIGHT_KEY = "light"

EXTENSIONS = {
        CAMERA_KEY: "yml",
        POSE_KEY: "txt",
        LIGHT_KEY: "txt",
    }

@FUNC.command()
def generate_dataset_train(
        path : Path = t.Option(..., help="Specify the folder containing light sets"),
        output_path: Path = t.Option(..., help="Specify the output underfolder path"),
        debug: bool = t.Option(False, help="Activate debug utilities")
        ):

    path: Path = Path(path)

    light_sets_uf = [UnderfolderReader(ls) for ls in sorted(path.iterdir()) if ls.is_dir()]

    num_cam = len(light_sets_uf[0])
    num_light = len(light_sets_uf)

    c_idxs = np.arange(0,num_cam)
    l_idxs = np.arange(0,num_light)
    l_idxs = np.tile(l_idxs,reps=math.ceil(num_cam/num_light))[:num_cam] 
    #l_idxs_b = np.repeat(l_idxs,num_repeated_lights)[:num_poses]   

    np.random.shuffle(c_idxs)
    np.random.shuffle(l_idxs)

    if debug:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    new_sequence = []
    for c_idx,l_idx in zip(c_idxs,l_idxs): 

        
        sample = light_sets_uf[l_idx][c_idx]

        intrinsics = sample[CAMERA_KEY]["intrinsics"]

        sample[CAMERA_KEY]["intrinsics"]["camera_matrix"][0][0] /= 2
        sample[CAMERA_KEY]["intrinsics"]["camera_matrix"][1][1] /= 2
        sample[CAMERA_KEY]["intrinsics"]["camera_matrix"][0][2] /= 2
        sample[CAMERA_KEY]["intrinsics"]["camera_matrix"][1][2] /= 2
        sample[CAMERA_KEY]["intrinsics"]["image_size"][0] = int(intrinsics["image_size"][0]/2)
        sample[CAMERA_KEY]["intrinsics"]["image_size"][1] = int(intrinsics["image_size"][1]/2)
        sample[IMAGE_KEY] = cv2.resize(sample[IMAGE_KEY], dsize=[int(i) for i in intrinsics["image_size"]], interpolation=cv2.INTER_AREA)
        new_sequence.append(sample)



        if debug:
            image = cv2.flip(sample[IMAGE_KEY], flipCode=0)

            cv2.imshow("img", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)


    template = light_sets_uf[0].get_reader_template()
    template.root_files_keys = [k for k in template.root_files_keys if k != "light"]

    reset = OperationResetIndices()

    dataset = SamplesSequence(new_sequence)

    writer = UnderfolderWriterV2(
        folder=output_path,
        reader_template=template,
        num_workers=-1,
    )

    writer(reset(dataset))

@FUNC.command()
def generate_dataset_test_light360( 
    input_folder: Path = t.Option(..., help="Specify the input underfolder path"),
    output_folder : Path = t.Option(..., help="Specify the output underfolder path"),
    zenit : int = t.Option(..., help="zenit of the cirle of lights on the dome"),
    num_poses : int = t.Option(..., help="num poses to generate"),
    camera_id : int = t.Option(None, help="camera point of view from train poses")
): 

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
    
    writer = UnderfolderWriter(
        folder = output_folder,
        root_files_keys=[CAMERA_KEY],
        extensions_map=EXTENSIONS,
    )

    if camera_id is None : camera_id = np.random.randint(low=0, high=len(camera_poses))
    
    camera_metadata = uf[0][CAMERA_KEY]

    samples = []
    for idx in range(num_poses):

        data = {
            CAMERA_KEY: camera_metadata,
            POSE_KEY: camera_poses[camera_id], 
            LIGHT_KEY: test_light_poses[idx]
        }
        sample = PlainSample(data=data, id=idx)
        samples.append(sample)

    writer(SamplesSequence(samples))
 

    

def generate_dataset_test_movingCamera( 
    input_folder: Path = t.Option(..., help="Specify the input underfolder path"),
    output_folder : Path = t.Option(..., help="Specify the output underfolder path"),
    zenit : int = t.Option(..., help="zenit of the cirle of lights on the dome"),
    num_poses : int = t.Option(..., help="num poses to generate"),
    light_id : int = t.Option(None, help="light position from train poses")
): 

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
    
    writer = UnderfolderWriter(
        folder = output_folder,
        root_files_keys=[CAMERA_KEY],
        extensions_map=EXTENSIONS,
    )

    if camera_id is None : camera_id = np.random.randint(low=0, high=len(camera_poses))
    
    camera_metadata = uf[0][CAMERA_KEY]

    samples = []
    for idx in range(num_poses):

        data = {
            CAMERA_KEY: camera_metadata,
            POSE_KEY: camera_poses[camera_id], 
            LIGHT_KEY: test_light_poses[idx]
        }
        sample = PlainSample(data=data, id=idx)
        samples.append(sample)

    writer(SamplesSequence(samples))
 


if __name__=="__main__":
    FUNC()