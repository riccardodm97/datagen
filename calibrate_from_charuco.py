import os 
from os import listdir
from statistics import mean

import yaml
from pathlib import Path
import cv2
from cv2 import aruco 
import numpy as np 
import argparse
from dotmap import DotMap


def load_images(path: str) -> list:
    
    img_files = [f for f in listdir(path) if f.endswith(".jpg")]
    for file in img_files:
        file = path / file
        file.rename(file.parent / (file.stem.zfill(5) + file.suffix))

    files = [f for f in listdir(path) if f.endswith(".jpg")]
    files = sorted(files)

    return files

def calibrate(config_file : str, debug : bool):

    assert os.path.exists(config_file), 'config yaml file not found'

    with open(config_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = DotMap(cfg_dict,_dynamic=False)

    data_path = Path(cfg.calib_path)
    n_markers = cfg.dictionary.n_markers
    marker_res = cfg.dictionary.marker_res
    square_size_meters = cfg.square_size_meter
    marker_size_meters = cfg.marker_size_meter
    grid_w = cfg.grid_size[1]
    grid_h = cfg.grid_size[0]


    # n_markers = 18
    # marker_res = 5
    # grid_size = [6,6]
    # square_size_meters = 0.03
    # marker_size_meters = 0.018
    # grid_w = grid_size[1]
    # grid_h = grid_size[0]


    aruco_dict = aruco.Dictionary_create(n_markers, marker_res)
    #aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    board = cv2.aruco.CharucoBoard_create(
            grid_w,
            grid_h,
            square_size_meters,
            marker_size_meters,
            aruco_dict,
        )

    all_corners = []
    all_ids = []

    images = load_images(data_path)

    for img_name in images : 

        img = cv2.imread(str(data_path / img_name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict)

        if debug : 
            img = aruco.drawDetectedMarkers(image=img, corners=corners, ids= ids)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            

        corners, ids, _, _ = aruco.refineDetectedMarkers(
            img_gray,
            board,
            corners,
            ids,
            rejected,
            cameraMatrix=None,
            distCoeffs=None,
        )

        n, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners,
            ids,
            img_gray,
            board,
            minMarkers=1,
            cameraMatrix=None,
            distCoeffs=None,
        )


        if n>0 :
            if debug : 
                img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                cornerColor=(100, 0, 255))

                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                cv2.imshow('Image', img)
                cv2.waitKey(0)
            
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

    
    img_size = img_gray.shape[1], img_gray.shape[0]

    intrinsics_init = np.array([
        [1.0, 0.0, img_size[0] / 2.0],
        [0.0, 1.0, img_size[1] / 2.0],
        [0.0, 0.0, 1.0]])

    distCoeff_init = np.zeros((5, 1))

    # Calibration routine
    calibration = aruco.calibrateCameraCharucoExtended(
        all_corners,
        all_ids,
        board,
        img_size,
        intrinsics_init,
        distCoeff_init,
        flags=0,
    )

    if debug : 
        for idx,img_name in enumerate(images) : 

            img = cv2.imread(str(data_path / img_name))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict)

            if debug : 
                img = aruco.drawDetectedMarkers(image=img, corners=corners, ids= ids)
                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                cv2.imshow('Image', img)
                cv2.waitKey(0)
                

            corners, ids, _, _ = aruco.refineDetectedMarkers(
                img_gray,
                board,
                corners,
                ids,
                rejected,
                cameraMatrix=None,
                distCoeffs=None,
            )

            n, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners,
                ids,
                img_gray,
                board,
                minMarkers=1,
                cameraMatrix=None,
                distCoeffs=None,
            )
            
            ret, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners,charuco_ids, board, calibration[1], 
                calibration[2], calibration[3][idx], calibration[4][idx], useExtrinsicGuess = False)
            img = cv2.drawFrameAxes(img, calibration[1], calibration[2], rvec, tvec, length=0.1)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow('Image', img)
            cv2.waitKey(0)


    camera_metadata = {
        "camera_pose": {
            "rotation": np.eye(3).tolist(),
            "translation": np.zeros(3).tolist(),
        },
        "intrinsics": {
            "camera_matrix": calibration[1].tolist(),
            "dist_coeffs": calibration[2].tolist(),
            "image_size": list(img_size),
        },
    }

    print(calibration[1])
    print(np.mean(calibration[-1]))

    with open(data_path/ 'camera.yml', "w") as f : 
        yaml.dump(camera_metadata, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_file', type=str, help='path to yml config file', required=True)
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')

    args = parser.parse_args()
    
    calibrate(args.config_file,args.debug)






# def calibrate_image(
#     input_path: Path = t.Option(..., help="Input image"),
#     n_markers: int = t.Option(..., help="Number of markers in the board"),
#     marker_bits: int = t.Option(..., help="Number of bits used to generate marker (e.g. 4, 5, ...)"),
#     grid_w: int = t.Option(..., help="Number of markers along X direction"),
#     grid_h: int = t.Option(..., help="Number of markers along Y direction"),
#     square_size_meters: float = t.Option(..., help="Size in meters of the outer square of the board"),
#     marker_size_meters: float = t.Option(..., help="Size in meters of the inner aruco marker"),
#     offset: int = t.Option(..., help="Specify id offset"),
#     debug: bool = t.Option(False, help="Activate debug mode")
# ):

#     input_image = iio.imread(input_path)
#     dictionary = cv2.aruco.Dictionary_create(n_markers, marker_bits)
#     board = cv2.aruco.CharucoBoard_create(grid_w, grid_h, square_size_meters, marker_size_meters, dictionary)
#     if debug:
#         input_image = cv2.cvtColor(board.draw((500, 500)), cv2.COLOR_GRAY2BGR)
#         cv2.imwrite("/tmp/marker.png", cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

#     corners, ids, rejected_points = cv2.aruco.detectMarkers(input_image, dictionary)
#     ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids - offset, input_image, board)
#     cv2.aruco.drawDetectedCornersCharuco(input_image, c_corners, c_ids, cornerColor=(100, 0, 255))

#     rich.print(corners)

#     cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#     cv2.imshow('Image', input_image)

#     cv2.waitKey(0)

# if __name__=="__main__":
#     t.run(calibrate_image)