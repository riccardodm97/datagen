import os
import shutil
from os import listdir
from pathlib import Path

import cv2
from pipelime.sequences.readers.filesystem import UnderfolderReader


def duplicate_file_in_underfolder(from_file : str, dest_folder : str, file_name : str):

    dest_folder = Path(dest_folder)

    uf = UnderfolderReader(dest_folder)
    for idx,sample in enumerate(uf):
        dest_path = dest_folder / 'data' / (str(idx).zfill(2) + f'_{file_name}')     
        shutil.copy(from_file,dest_path)


#duplicate_file_in_underfolder('/home/eyecan/dev/real_relight/data/datasets/train/cubesOnShelf_multiLight/uf/data/018_pose.txt',
#'/home/eyecan/dev/real_relight/data/datasets/test/cubesOnShelf_multiLight/light180_2/uf', 'pose.txt')



def resize_imgs_underfolder(img_folder : str):

    images_path = Path(img_folder)

    width = 696
    height = 464
    dim = (width, height)

    def load_images(path: str) -> list:

        files = [f for f in listdir(path) if f.endswith(".jpeg")]
        files = sorted(files)

        return files


    images = load_images(images_path)

    for img_name in images : 

        img_path = str(images_path / img_name)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        os.remove(img_path)

        cv2.imwrite(img_path, resized)
    
    print('Resized Dimensions : ',resized.shape)



def rename_file_underfolder(uf_datapath : str, rename_to : str):

    uf_datapath = Path(uf_datapath)

    img_files = [f for f in listdir(uf_datapath) if f.endswith(".jpg")]
    img_files = sorted(img_files)

    for idx,file in enumerate(img_files):
        file = uf_datapath / file
        file.rename(file.parent / (str(idx).zfill(2) + f'_{rename_to}' + file.suffix.lower()))   

    img_files = [f for f in listdir(uf_datapath) if f.endswith(".jpg")]
    img_files = sorted(img_files)
    
    print(img_files)     
