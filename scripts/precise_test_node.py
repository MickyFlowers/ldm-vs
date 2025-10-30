import cv2
import numpy as np
import argparse
import time
from scipy.spatial.transform import Rotation as R
from xlib.device.sensor.camera import RealSenseCamera
from xlib.algo.cv.detector import *
from xlib.algo.utils.transforms import matrixToVec
from xlib.log import logging
from collections import deque


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
    aruco_detector_config = {
        "aruco_type": cv2.aruco.DICT_6X6_250,
        "marker_size": 0.015,
        "id": 25,
    }
    target_pose_mtx = np.eye(4)
    target_pose_mtx = np.load("data/test/expect_pose_screw_m.npy")
    count = 0
    rvec_queue = deque(maxlen=100)
    tvec_queue = deque(maxlen=100)
    while True:
        color, _ = camera.get_frame()
        image, mtx = get_single_aruco_pose(
            aruco_detector_config,
            color,
            camera.intrinsics_matrix,
            camera.distortion,
        )
        if mtx is not None:
            rvec, tvec = matrixToVec(mtx)
            rvec_queue.append(rvec)
            tvec_queue.append(tvec)
        # if count == 0:
        #     image, mtx = get_single_aruco_pose(
        #         aruco_detector_config,
        #         color,
        #         camera.intrinsics_matrix,
        #         camera.distortion,
        #     )
        #     if mtx is not None:
        #         rvec, tvec = matrixToVec(mtx)
        #         count += 1
        # else:
        #     image, mtx = get_single_aruco_pose(
        #         aruco_detector_config,
        #         color,
        #         camera.intrinsics_matrix,
        #         camera.distortion,
        #         rvec,
        #         tvec,
        #     )
        #     if mtx is not None:
        #         rvec, tvec = matrixToVec(mtx)

        if len(rvec_queue) == rvec_queue.maxlen and len(tvec_queue) == tvec_queue.maxlen:
            rvec = np.mean(rvec_queue, axis=0).reshape(-1)
            rvec_std = np.std(rvec_queue, axis=0).reshape(-1)
            tvec = np.mean(tvec_queue, axis=0).reshape(-1)
            tvec_std = np.std(tvec_queue, axis=0).reshape(-1)
            # mtx = vecToMatrix(rvec, tvec)
            delta_mtx = np.linalg.inv(target_pose_mtx) @ mtx
            delta_rot_mtx = delta_mtx[:3, :3]
            delta_rot_vec = R.from_matrix(delta_rot_mtx).as_rotvec()
            delta_vec = delta_mtx[:3, 3]
            print(f"rvec: {rvec}; rvec std: {rvec_std}")
            print(f"tvec: {tvec}; tvec std: {tvec_std}")
            print(
                f"delta_rot_vec: {delta_rot_vec}; delta vec: {delta_vec}; delta rot norm: {np.linalg.norm(delta_rot_vec)}; delta vec norm: {np.linalg.norm(delta_vec)}"
            )
        cv2.imshow("aruco pose", image)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            target_pose_mtx = mtx

            np.save("data/test/expect_pose_screw_m.npy", mtx)
        time.sleep(1.0 / 20.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--serial_number", type=str, default="f1421776")
    parser.add_argument("--serial_number", type=str, default="408122071109")  # d435
    parser.add_argument("--exposure_time", type=int, default=400)
    args = parser.parse_args()
    main(args)
