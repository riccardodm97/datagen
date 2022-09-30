import numpy as np
import cv2


def variance_of_laplacian(image):
    """Compute the variance of the Laplacian of the given image.

    :param image: input image
    :type image: _type_
    :return: variance of laplacian
    :rtype: _type_
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    """Compute the sharpness of the given image.

    :param image: input image
    :type image: _type_
    :return: sharpness
    :rtype: _type_
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def rotate_x_axis(pose: np.ndarray) -> np.ndarray:
    """Rotate a pose of 180 degrees around the x-axis

    :param pose: input pose
    :type pose: np.ndarray
    :return: output pose
    :rtype: np.ndarray
    """

    # Rotation of 180 degrees along x axis
    rx = np.eye(4).astype(np.float32)
    rx[1, 1] = rx[2, 2] = -1
    pose = np.dot(pose, rx)
    return pose


# def rotate_x_axis(pose: np.ndarray) -> np.ndarray:
#     """Converts a camera pose to/from the blender reference frame
#     (rotated of 180 degrees along x axis).

#     :param pose: the input pose
#     :type pose: np.ndarray
#     :return: the converted pose
#     :rtype: np.ndarray
#     """
#     from transforms3d import affines, euler

#     t, r, _, _ = affines.decompose(pose)
#     r_euler = np.array(euler.mat2euler(r))
#     r_euler[0] += np.pi
#     r = euler.euler2mat(*r_euler)
#     new_pose = np.eye(4)
#     new_pose[:3, :3] = r
#     new_pose[:3, 3] = t
#     return new_pose


# def generate_charuco_board():
#     import cv2 
#     from cv2 import aruco 

#     n_markers = 18
#     marker_res = 5
#     grid_size = [6,6]
#     square_size_meters = 0.03
#     marker_size_meters = 0.018

#     dictionary = aruco.Dictionary_create(n_markers, marker_res)
#     board = aruco.CharucoBoard_create(grid_size[1], grid_size[0], square_size_meters, marker_size_meters, dictionary)

#     input_image = cv2.cvtColor(board.draw((500, 500)), cv2.COLOR_GRAY2BGR)
#     cv2.imwrite("/home/eyecan/dev/real_relight/data/datasets/train/prove/prova0/calibration/board.jpg", cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))