import os 
import argparse
import yaml
from dotmap import DotMap
from pathlib import Path
from os import listdir
from os.path import isfile
import h5py 
import numpy as np
import pickle
import shutil

from pipelime.sequences.writers.filesystem import UnderfolderWriter
from pipelime.sequences.samples import PlainSample, SamplesSequence


BASE_PATH = Path('~/dev').expanduser()      #TOCHANGE cosi è specifico per questo pc 
DATASET_PATH = BASE_PATH/'data'/'dataset'
BLENDER_PATH = BASE_PATH/'data'/'blender_tmp'


IMAGE_KEY = 'image'
DEPTH_KEY = 'depth'
LIGHT_KEY = 'light'
POSE_KEY = 'pose'
CAMERA_KEY = 'camera'
BBOX_KEY = 'bbox'

EXTENSIONS = {
        IMAGE_KEY: "png",
        DEPTH_KEY: "npy",
        LIGHT_KEY: "txt",
        CAMERA_KEY: "yml",
        POSE_KEY: "txt",
        BBOX_KEY: "txt",
    }

# load all rendered files from a directory
def load_rendered_files(path: str) -> list:
  
    for file in listdir(path):
        file = path / file
        file.rename(file.parent / (file.stem.zfill(5) + file.suffix))

    render_files = [f for f in listdir(path) if isfile(path / f)]
    render_files = sorted(render_files)

    return render_files

def main(dataset_id : str) :

    cfg_file = os.path.join(BLENDER_PATH,dataset_id,'metadata.yml')
    assert os.path.exists(cfg_file), 'config yaml file not found'

    with open(cfg_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotMap(cfg_dict,_dynamic=False)
    
    folder = 'train' if cfg.train else f'test/{cfg.consistent_with}'  #DEBUG

    in_blender_tmp_folder = BLENDER_PATH / dataset_id
    out_underfolder_path = DATASET_PATH / folder / dataset_id
    
    render_path = in_blender_tmp_folder / 'render'
    c_poses_path = in_blender_tmp_folder / 'c_poses.npy'
    l_poses_path = in_blender_tmp_folder / 'l_poses.npy'
    bbox_path = in_blender_tmp_folder / 'bbox.npy'
    camera_metadata_path = in_blender_tmp_folder / 'camera_metadata.pickle'


    writer = UnderfolderWriter(
        folder= out_underfolder_path,
        root_files_keys=[CAMERA_KEY, BBOX_KEY],
        extensions_map=EXTENSIONS,
    )

    #load files from blender folder 
    camera_poses = np.load(c_poses_path)
    light_poses = np.load(l_poses_path)
    bbox = np.load(bbox_path)

    with open(camera_metadata_path, 'rb') as m:
        camera_metadata = pickle.load(m)
        
    render_files = load_rendered_files(render_path)

    samples = []
    for idx, file_h5py in enumerate(render_files):
        file_h5py = h5py.File(render_path / file_h5py, mode="r")
        rgb = file_h5py["colors"][:]
        depth = file_h5py["depth"][:]

        data = {
            IMAGE_KEY: rgb.astype(np.uint8),
            DEPTH_KEY: depth,
            CAMERA_KEY: camera_metadata,
            BBOX_KEY: bbox,
            POSE_KEY: camera_poses[idx], 
            LIGHT_KEY: light_poses[idx]
        }
        sample = PlainSample(data=data, id=idx)
        samples.append(sample)

    writer(SamplesSequence(samples))

    try : 
        shutil.copy(cfg_file, out_underfolder_path)                # copy yaml cfg file to out directory 
        shutil.move(str(render_path), str(out_underfolder_path))   # move render folder from tmp blender one to dataset folder
        shutil.rmtree(in_blender_tmp_folder)                       # completely delete blender tmp folder 
    except Exception as e: 
        print(str(e))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', dest='dataset_id', type=str, help='id of the dataset', required=True)
    
    args = parser.parse_args()
    
    main(args.dataset_id)