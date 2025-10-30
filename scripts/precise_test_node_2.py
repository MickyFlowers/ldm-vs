import cv2
import numpy as np
import argparse
import time
from scipy.spatial.transform import Rotation as R
from xlib.device.sensor.camera import RealSenseCamera
from xlib.algo.cv.detector import *


def main(args):
    camera = RealSenseCamera(
        serial_number=args.serial_number, exposure_time=args.exposure_time
    )

    def camera_warmup(warm_up_steps=10):
        for step in range(warm_up_steps):
            camera.get_frame()
            time.sleep(1.0 / 30.0)
            print(f"warming up camera: step: {step:03d}")

    camera_warmup()
    aruco_detector_config = {"aruco_type": cv2.aruco.DICT_6X6_250, "marker_size": 0.015, "id": 25}
    aruco_detector_config_2 = {
        "aruco_type": cv2.aruco.DICT_6X6_50,
        "marker_size": 0.028,
        "num_markers": (2, 2),
        "marker_seperation": 0.004,
        "ids": [0, 1, 2, 3],
    }
    # target_pose_mtx = np.eye(4)
    target_pose_mtx = np.load("data/test/expect_pose_screw_m.npy", allow_pickle=True)

    while True:
        color, _ = camera.get_frame()
        image, mtx = get_single_aruco_pose(
            aruco_detector_config, color, camera.intrinsics_matrix, camera.distortion
        )
        image, mtx2 = get_aruco_pose(
            aruco_detector_config_2,
            image,
            camera.intrinsics_matrix,
            camera.distortion,
        )
        if mtx is not None and mtx2 is not None:
            relative_pose = np.linalg.inv(mtx2) @ mtx
            delta_mtx = np.linalg.inv(target_pose_mtx) @ relative_pose
            delta_rot_mtx = delta_mtx[:3, :3]
            delta_rot_vec = R.from_matrix(delta_rot_mtx).as_rotvec()
            delta_vec = delta_mtx[:3, 3]

            print(
                f"delta_rot_vec: {delta_rot_vec}; delta vec: {delta_vec}; delta rot norm: {np.linalg.norm(delta_rot_vec)}; delta vec norm: {np.linalg.norm(delta_vec)}"
            )
        cv2.imshow("aruco pose", image)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            target_pose_mtx = relative_pose

            np.save("data/test/expect_pose_screw_m.npy", relative_pose)
        time.sleep(1.0 / 20.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--serial_number", type=str, default="f1421776")
    parser.add_argument("--serial_number", type=str, default="408122071109")  # d435
    parser.add_argument("--exposure_time", type=int, default=500)
    args = parser.parse_args()
    main(args)
