import click
from utils.utils import rotate_x_axis
import math
import cv2
import numpy as np
from pathlib import Path
import os
from choixe.configurations import XConfig
from rich.progress import track
from pipelime.sequences.readers.filesystem import UnderfolderReader
from pipelime.sequences.operations import OperationShuffle, OperationSplits


def convert2nerf(uf, extension, output_folder, camera_key, image_key, pose_key, resize):

    output_folder = Path(output_folder)
    (output_folder / "images").mkdir(parents=True, exist_ok=True)

    if resize is not None:
        resize = int(resize)

        camera_matrix = np.array(camera["intrinsics"]["camera_matrix"]) / resize
        camera_matrix[2, 2] = 1

        camera = uf[0][camera_key]
        dist_coeffs = camera["intrinsics"]["dist_coeffs"]

        w, h = np.array(camera["intrinsics"]["image_size"]) / resize
        w, h = int(w), int(h)
    else:
        camera = uf[0][camera_key]
        camera_matrix = camera["intrinsics"]["camera_matrix"]
        dist_coeffs = camera["intrinsics"]["dist_coeffs"]
        w, h = camera["intrinsics"]["image_size"]

    fl_x = camera_matrix[0][0]
    fl_y = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]

    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    transform = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": dist_coeffs[0][0],
        "k2": dist_coeffs[1][0],
        "p1": dist_coeffs[2][0],
        "p2": dist_coeffs[3][0],
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": [],
    }

    for sample in track(uf):
        src_image_file = Path(sample.filesmap[image_key])
        dst_image_file = output_folder / "images" / src_image_file.name

        if resize is not None:
            img = cv2.imread(str(src_image_file), flags=cv2.IMREAD_UNCHANGED)
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            scaled_img = cv2.resize(img, [w, h])
            cv2.imwrite(str(dst_image_file), scaled_img)
        else:
            os.link(src_image_file, dst_image_file)

        pose = sample[pose_key]
        c2w = rotate_x_axis(pose)

        frame = {
            "file_path": f"images/{src_image_file.stem}",
            "transform_matrix": c2w,
        }
        transform["frames"].append(frame)
        save_name = "transforms_" + extension + ".json"
        XConfig.from_dict(transform).to_json(output_folder / save_name)


@click.command(help="Convert a underfolder to the nerf format")
@click.option(
    "-i", "--input_folder", type=str, required=True, help="Input train dataset"
)
@click.option("-o", "--output_folder", type=str, required=True, help="Output dataset")
@click.option("--camera_key", default="camera", help="Camera key")
@click.option("--image_key", default="image", help="Image key")
@click.option("--pose_key", default="pose", help="Pose key")
@click.option("--resize", default=None, help="resize scale (H/resize, W/resize)")
def underfolder2nerf(
    input_folder,
    output_folder,
    camera_key,
    image_key,
    pose_key,
    resize,
):

    uf = UnderfolderReader(input_folder)
    uf = OperationShuffle(seed=0)(uf)
    splits = OperationSplits({"train": 0.90, "val": 0.05, "test": 0.05})(uf)

    for split, uf in splits.items():
        print("Geneerating {} dataset".format(split))
        convert2nerf(uf, split, output_folder, camera_key, image_key, pose_key, resize)


if __name__ == "__main__":
    underfolder2nerf()
