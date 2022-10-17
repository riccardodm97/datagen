import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from pipelime.sequences.samples import PlainSample, SamplesSequence
from pipelime.sequences.writers.filesystem import UnderfolderWriter

TASK = 'real_relight' 
BASE_PATH = Path('~/dev').expanduser()      
TASK_PATH = BASE_PATH / TASK
DATASET_PATH = TASK_PATH / 'data' / 'datasets'


IMAGE_KEY = 'image'
LIGHT_KEY = 'light'
POSE_KEY = 'pose'
CAMERA_KEY = 'camera'

EXTENSIONS = {
        IMAGE_KEY: "png",
        LIGHT_KEY: "txt",
        CAMERA_KEY: "yml",
        POSE_KEY: "txt",
    }
    

def colmap2uf(dataset_id : str):

    data_folder_path = DATASET_PATH / dataset_id 
    image_folder_path = data_folder_path / 'images'
    underfolder_path = data_folder_path / 'uf'
    transforms_json_path = data_folder_path / 'tranforms.json'

    data = json.load(open(transforms_json_path, "r"))

    camera_matrix = [
        [data["fl_x"], data["cx"], 0.0],
        [0.0, data["fl_y"], data["cy"]],
        [0.0, 0.0, 1.0],
    ]
    camera_metadata = {
        "camera_pose": {
            "rotation": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
        },
        "intrinsics": {
            "camera_matrix": camera_matrix,
            "dist_coeffs": [data["k1"], data["k2"], data["p1"], data["p2"]],
            "image_size": [data["w"], data["h"]],
        },
    }

    writer = UnderfolderWriter(
        folder= underfolder_path,
        root_files_keys=[CAMERA_KEY],
        extensions_map=EXTENSIONS,
    )

    samples = []
    for idx,frame in enumerate(data["frames"]):
        img = imageio.imread(image_folder_path / frame["file_path"])
        pose = np.array(frame["transform_matrix"])
        pose[:, [1, 2]] = -pose[:, [1, 2]]

        data = {
            IMAGE_KEY: img,
            CAMERA_KEY: camera_metadata,
            POSE_KEY: pose, 
        }
        sample = PlainSample(data=data, id=idx)
        samples.append(sample)
    
    writer(SamplesSequence(samples))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', dest='dataset_id', type=str, help='id of the dataset', required=True)
    
    args = parser.parse_args()
    
    colmap2uf(args.dataset_id)