import blenderproc as bproc
import numpy as np
from pathlib import Path
import bpy
from os import listdir
from os.path import isfile
import h5py
import math
import click
from pipelime.sequences.writers.filesystem import UnderfolderWriter
from pipelime.sequences.samples import PlainSample, SamplesSequence
from transforms3d import affines, euler


def get_render_files(path: str) -> list:
    """_summary_

    :param path: _description_
    :type path: str
    :return: _description_
    :rtype: list
    """
    for file in listdir(path):
        file = path / file
        file.rename(file.parent / (file.stem.zfill(5) + file.suffix))

    render_files = [f for f in listdir(path) if isfile(path / f)]
    render_files = sorted(render_files)
    return render_files


def get_bbox(obj_bbox: str) -> np.array:
    """_summary_

    :param obj_bbox: _description_
    :type obj_bbox: str
    :return: _description_
    :rtype: np.array
    """
    scanbox_dimensions = bpy.data.objects[obj_bbox].dimensions
    bbox = np.array(
        [
            -scanbox_dimensions.x / 2,
            -scanbox_dimensions.y / 2,
            0.0,
            scanbox_dimensions.x / 2,
            scanbox_dimensions.y / 2,
            scanbox_dimensions.z,
        ]
    )
    return np.round(bbox, 4)


def convert_to_rgb(file_h5py: dict) -> np.array:
    """_summary_

    :param file_h5py: _description_
    :type file_h5py: dict
    :return: _description_
    :rtype: np.array
    """
    return file_h5py["colors"][:]


def convert_to_depth(file_h5py: dict) -> np.array:
    """_summary_

    :param file_h5py: _description_
    :type file_h5py: dict
    :return: _description_
    :rtype: np.array
    """
    return file_h5py["depth"][:]


def generate_dome_camera_rotate(
    objs: list, t_matrix: np.array, r: float, N: int
) -> tuple:
    """_summary_

    :param objs: _description_
    :type objs: list
    :param t_matrix: _description_
    :type t_matrix: np.array
    :param r: _description_
    :type r: float
    :param N: _description_
    :type N: int
    :return: _description_
    :rtype: tuple
    """
    poi = bproc.object.compute_poi(objs)

    poses = []
    poses_lights = []

    for i in range(N):
        location = bproc.sampler.part_sphere(
            center=poi,
            radius=r,
            part_sphere_dir_vector=[0, 0, 1],
            mode="SURFACE",
            dist_above_center=0.0,
        )
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )
        poses.append(cam2world_matrix)

        c2l = np.matmul(cam2world_matrix, t_matrix)
        light_location = np.array([c2l[0, 3], c2l[1, 3], c2l[2, 3]])
        light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
        cam2lightworld_matrix = bproc.math.build_transformation_mat(
            light_location, light_rotation
        )

        poses_lights.append(cam2lightworld_matrix)

    return poses, poses_lights


def generate_single_zenith(
    objs: list, t_matrix: np.array, z: int, r: float, N: int
) -> tuple:
    """_summary_

    :param objs: _description_
    :type objs: list
    :param t_matrix: _description_
    :type t_matrix: np.array
    :param z: _description_
    :type z: int
    :param r: _description_
    :type r: float
    :param N: _description_
    :type N: int
    :return: _description_
    :rtype: tuple
    """
    poi = bproc.object.compute_poi(objs)

    poses = []
    poses_lights = []

    z = math.radians(z)

    for i in range(N):
        poi_ = poi.copy()
        poi_[2] += r * math.cos(z)
        location = bproc.sampler.disk(
            center=poi_,
            radius=r * math.sin(z),
            # rotation=0,
            sample_from="circle",
        )
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )
        poses.append(cam2world_matrix)

        c2l = np.matmul(cam2world_matrix, t_matrix)
        light_location = np.array([c2l[0, 3], c2l[1, 3], c2l[2, 3]])
        light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
        cam2lightworld_matrix = bproc.math.build_transformation_mat(
            light_location, light_rotation
        )
        poses_lights.append(cam2lightworld_matrix)

    return poses, poses_lights


def generate_test(objs: list, t_matrix: np.array, r: float, N: int) -> tuple:
    """_summary_

    :param objs: _description_
    :type objs: list
    :param t_matrix: _description_
    :type t_matrix: np.array
    :param r: _description_
    :type r: float
    :param N: _description_
    :type N: int
    :return: _description_
    :rtype: tuple
    """
    poi = bproc.object.compute_poi(objs)

    def points_on_circle(r, n=100, z=1):
        return [
            [math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r, z]
            for x in range(0, n + 1)
        ]

    poses = []
    poses_lights = []

    z = math.radians(45)

    poi_ = poi.copy()
    poi_[2] += r * math.cos(z)
    location = bproc.sampler.disk(
        center=poi_,
        radius=r * math.sin(z),
        # rotation=0,
        sample_from="circle",
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # compute a light trajectory
    lights_positions = points_circle(r, N, 0.5)
    for position in lights_positions:
        poses.append(cam2world_matrix)

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - position)
        c2w = bproc.math.build_transformation_mat(position, rotation_matrix)
        c2l = np.matmul(c2w, t_matrix)
        light_location = np.array([c2l[0, 3], c2l[1, 3], c2l[2, 3]])
        light_rotation = bproc.camera.rotation_from_forward_vec(poi - light_location)
        cam2lightworld_matrix = bproc.math.build_transformation_mat(
            light_location, light_rotation
        )
        poses_lights.append(cam2lightworld_matrix)

    return poses, poses_lights

def points_circle(r, n=100, z=1):
        return [
            [math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r, z]
            for x in range(0, n + 1)
        ]

def points_on_circumference(center: Tuple[float, float] = (0.0, 0.0), r: float = 50.0, n: int = 100) -> List[Tuple[float, float]]:
    return [
        (
            center[0] + (math.cos(2 * math.pi / n * x) * r),  # x
            center[1] + (math.sin(2 * math.pi / n * x) * r),  # y
        )
        for x in range(0, n + 1)]


# The lights are attached to the camera and not imported from the scene
@click.command(
    help="Generate a sequence of samples from a given scene, the light attached to the camera"
)
@click.option("-s", "--scene", type=str, required=True, help=".blend file")
@click.option("-o", "--output_folder", type=str, required=True, help="Output folder")
@click.option(
    "--number_images", default=50, type=int, help="Number of images to generate"
)
@click.option(
    "--radius", default=0.65, type=float, help="Radius of the camera in meters"
)
@click.option(
    "--n_zenith",
    default=5,
    type=int,
    help="Number of zeniths",
)
@click.option("--width", default=360, type=int, help="Image width")
@click.option("--height", default=270, type=int, help="Image height")
@click.option(
    "--obj_bbox", default="Cube", type=str, help="Object from where compute the bbox"
)
@click.option("--image_key", default="image", type=str, help="Image key")
@click.option("--depth_key", default="depth", type=str, help="depth key")
@click.option("--light_key", default="light", type=str, help="light key")
@click.option("--pose_key", default="pose", type=str, help="Pose key")
@click.option("--camera_key", default="camera", type=str, help="camera key")
@click.option("--bbox_key", default="bbox", type=str, help="bbox key")
@click.option(
    "--render_dome", is_flag=True, default=False, help="Render a dome of radius"
)
@click.option(
    "--render_zenith", is_flag=True, default=False, help="Render n zenith of radius"
)
@click.option(
    "--render_test",
    is_flag=True,
    default=False,
    help="Render the test set",
)
def main(
    scene,
    output_folder,
    number_images,
    radius,
    n_zenith,
    width,
    height,
    obj_bbox,
    image_key,
    depth_key,
    light_key,
    pose_key,
    camera_key,
    bbox_key,
    render_dome,
    render_zenith,
    render_test,
):

    extensions = {
        image_key: "png",
        depth_key: "npy",
        light_key: "txt",
        camera_key: "yml",
        pose_key: "txt",
        bbox_key: "txt",
    }

    writer = UnderfolderWriter(
        folder=Path(output_folder) / "underfolder",
        root_files_keys=[camera_key, bbox_key],
        extensions_map=extensions,
    )

    bproc.init()
    # Transparent Background
    bproc.renderer.set_output_format(enable_transparency=True)
    # Import just the objects and not the lights
    objs = bproc.loader.load_blend(scene, data_blocks=["objects"], obj_types=["mesh"])
    bproc.camera.set_resolution(width, height)

    camera_metadata = {
        "camera_pose": {
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 0],
        },
        "intrinsics": {
            "camera_matrix": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
            "dist_coeffs": np.zeros((5, 1)).tolist(),
            "image_size": [width, height],
        },
    }

    bbox = get_bbox(obj_bbox)

    t_matrix = np.eye(4)
    t_matrix[0, 3] = 2
    # t_matrix[1, 3] = radius * 2
    t_matrix[1, 3] = 0
    t_matrix[2, 3] = 0
    if render_dome:
        poses, poses_lights = generate_dome_camera_rotate(
            objs, t_matrix, radius, number_images
        )
    elif render_test:

        poses, poses_lights = generate_test(objs, t_matrix, radius, number_images)

    elif render_zenith:
        poses = []
        poses_lights = []
        zenith_step = 90 / n_zenith
        zenith = 0
        for _ in range(n_zenith):
            zenith += zenith_step
            poses_zenith, poses_lights_zenith = generate_single_zenith(
                objs, t_matrix, zenith, radius, int(number_images / n_zenith)
            )
            poses = poses + poses_zenith
            poses_lights = poses_lights + poses_lights_zenith

    else:
        raise NotImplementedError("Select render_dome")

    bproc.renderer.set_noise_threshold(16)
    bproc.renderer.enable_depth_output(False)

    light = bproc.types.Light()
    light.set_type("SPOT")
    light.set_energy(1000)
    for i in range(number_images):
        bproc.utility.reset_keyframes()
        bproc.camera.add_camera_pose(poses[i])
        t, r, _, _ = affines.decompose(poses_lights[i])
        r_euler = euler.mat2euler(r)
        light.set_location(t)
        light.set_rotation_euler(r_euler)
        data = bproc.renderer.render()
        data.update(
            bproc.renderer.render_segmap(
                map_by=["cp_object", "name", "instance"],
                default_values={"cp_object": 0},
            )
        )
        bproc.writer.write_hdf5(
            Path(output_folder) / "render", data, append_to_existing_output=True
        )

    render_files = get_render_files(Path(output_folder) / "render")

    samples = []
    for idx, file_h5py in enumerate(render_files):
        file_h5py = h5py.File(Path(output_folder) / "render" / file_h5py, mode="r")
        rgb = convert_to_rgb(file_h5py)
        depth = convert_to_depth(file_h5py)

        # light_position = [
        #     poses_lights[idx][0, 3],
        #     poses_lights[idx][1, 3],
        #     poses_lights[idx][2, 3],
        # ]

        data = {
            image_key: rgb.astype(np.uint8),
            pose_key: bproc.math.change_source_coordinate_frame_of_transformation_matrix(
                poses[idx], ["X", "-Y", "-Z"]
            ),
            camera_key: camera_metadata,
            bbox_key: bbox,
            light_key: bproc.math.change_source_coordinate_frame_of_transformation_matrix(
                poses_lights[idx], ["X", "-Y", "-Z"]
            ),
            # light_key: light_position,
            depth_key: depth,
        }
        sample = PlainSample(data=data, id=idx)
        samples.append(sample)

    writer(SamplesSequence(samples))


if __name__ == "__main__":
    main()
