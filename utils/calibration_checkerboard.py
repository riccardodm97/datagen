import argparse
from os import listdir
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_images(path: str) -> list:
    
    img_files = [f for f in listdir(path) if f.endswith(".JPG")]
    for file in img_files:
        file = path / file
        file.rename(file.parent / (file.stem.zfill(5) + file.suffix))

    files = [f for f in listdir(path) if f.endswith(".JPG")]
    files = sorted(files)

    return files


def detectCorners(
    path_image: Path, size_pattern: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    img = cv2.imread(str(path_image), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Failed to load {path_image}.")
        return None

    found, corners = cv2.findChessboardCorners(img, size_pattern)
    print(f"Found: {len(corners)} corners.")

    if found:
        # Refining corner position to subpixel iteratively until criteria  max_count=30 or criteria_eps_error=1 is sutisfied
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        # Image Corners
        corners_ref = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        # # Draw and display the corners
        # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.drawChessboardCorners(vis, size_pattern, corners, found)
        # plt.figure(figsize=(20, 10))
        # plt.imshow(vis)
        # plt.show()

    if not found:
        print("chessboard not found")
        return None

    return corners_ref.reshape(-1, 2)


def main(img_dir : str):

    # img_dir = Path(img_dir)
    # paths_board = load_images(img_dir)

    dir_img = Path("/home/eyecan/dev/real_relight/data/datasets/train/prove/prova1/calib_cam")
    paths_board = list(dir_img.glob("*"))
    paths_board.sort()
    
    print(paths_board)

    size_square = 26.5  # mm
    size_pattern = (8, 5)  # number of inner corner

    # Building 3D points
    indices = np.indices(size_pattern, dtype=np.float32)
    indices *= size_square
    pattern_points = np.zeros([size_pattern[0] * size_pattern[1], 3], np.float32)
    coords_3D = indices.T.reshape(-1, 2)
    pattern_points[:, :2] = coords_3D

    # Building 2D-3D correspondeces
    chessboards = [detectCorners(path_image, size_pattern) for path_image in paths_board]
    chessboards = [x for x in chessboards if x is not None]

    obj_points = []  # 3D points
    img_points = []  # 2D points

    for corners in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # Getting the width and height of the images
    h, w = cv2.imread(str(paths_board[0]), cv2.IMREAD_GRAYSCALE).shape[:2]
    print("Camera resolution w:%d h:%d" % (w, h))

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    print("rms :\n", rms)
    print("A_s :\n", camera_matrix)

    # Draw Cube
    for idx_image, path in enumerate(paths_board[:1]):
        img = cv2.imread(str(path))  # load the correct image

        red = (0, 0, 255)  # red (in BGR)
        blue = (255, 0, 0)  # blue (in BGR)
        green = (0, 255, 0)  # green (in BGR)
        line_width = 10

        corners_cube_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [0, 100, 0],
                [100, 100, 0],
                [100, 0, 0],
                [0, 0, -100],
                [0, 100, -100],
                [100, 100, -100],
                [100, 0, -100],
            ]
        )

        cube_corners_2d, _ = cv2.projectPoints(
            corners_cube_3d, rvecs[idx_image], tvecs[idx_image], camera_matrix, dist_coefs
        )
        cube_corners_2d = cube_corners_2d.astype(np.int32)
        # first draw the base in red
        cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0]), red, line_width)
        cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0]), red, line_width)
        cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0]), red, line_width)
        cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0]), red, line_width)

        # now draw the pillars
        cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0]), blue, line_width)
        cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0]), blue, line_width)
        cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0]), blue, line_width)
        cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0]), blue, line_width)

        # finally draw the top
        cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0]), green, line_width)
        cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0]), green, line_width)
        cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0]), green, line_width)
        cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0]), green, line_width)

        plt.imshow(img[..., ::-1])
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', dest='images_path', type=str, help='path to images', required=True)
    args = parser.parse_args()
    main(args.images_path)
