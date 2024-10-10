from xlib.device.manipulator.ur_robot import UR
from xlib.device.robotiq.robotiq_gripper import RobotiqGripper
from xlib.device.sensor.camera import RealSenseCamera
from xlib.algo.utils.random import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading
import cv2
import time

right_camera_to_tcp = np.load("data/vs_examples/extrinsic/right_camera_to_tcp.npy")
left_camera_to_tcp = np.load("data/vs_examples/extrinsic/left_camera_to_tcp.npy")
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
right_base_to_world = np.load("data/vs_examples/extrinsic/right_base_to_world.npy")
print(left_base_to_world)
print(right_base_to_world)

enable_gripper = False

left_ip = "10.51.33.233"
right_ip = "10.51.33.232"

# print(left_base_to_world)
left_robot = UR(left_ip, left_base_to_world)
right_robot = UR(right_ip, right_base_to_world)
camera = RealSenseCamera(exposure_time=450)
print(left_robot.world_pose)
print(right_robot.world_pose)

key = input("activate gripper? (y/n): ")
if key.lower() == "y":
    enable_gripper = True
    left_gripper = RobotiqGripper()
    left_gripper.connect(left_ip, 63352)
    left_gripper.activate()

left_arm_pose = np.eye(4)
left_arm_pose[:3, 3] = np.array([-1.51300577e-02, 6.15169458e-01, 4.33593506e-01])
left_arm_pose[:3, :3] = R.from_euler("XYZ", [np.pi, 0, -np.pi / 2]).as_matrix()
left_robot.moveToWorldPose(left_arm_pose, 0.1, 0.1)

jnt_error = [0, 0, 0, 0, 0, 0]
jnt_error[2] = np.pi / 180 * 3.5

if enable_gripper:
    key = input("Close gripper? (y/n): ")
    if key.lower() == "y":
        left_gripper.move_and_wait_for_pos(255, 255, 100)

in_hand_error_pos_lower = [-0.01, -0.03, -0.03]
in_hand_error_pos_upper = [0.01, 0.0, 0.02]
in_hand_error_rot_lower = [-45, -10, -30]
in_hand_error_rot_upper = [-20, 10, 30]
tip_to_tcp = np.eye(4)
tip_to_tcp[2, 3] = 0.16
count = 0
while count < 1e4:
    in_hand_error_pos = np.random.uniform(
        in_hand_error_pos_lower, in_hand_error_pos_upper
    )
    in_hand_error_rot = np.random.uniform(
        in_hand_error_rot_lower, in_hand_error_rot_upper
    )
    # test
    in_hand_error_trans_matrix = np.eye(4)
    in_hand_error_trans_matrix[:3, :3] = R.from_euler(
        "XYZ", in_hand_error_rot, degrees=True
    ).as_matrix()
    in_hand_error_trans_matrix[:3, 3] = in_hand_error_pos
    object_pose = left_arm_pose @ tip_to_tcp
    tip_pose = object_pose @ np.linalg.inv(in_hand_error_trans_matrix)
    tcp_pose = tip_pose @ np.linalg.inv(tip_to_tcp)
    trans = np.eye(4)
    trans[:3, :3] = R.from_euler("XYZ", [0, 0, np.pi]).as_matrix()
    back_tcp_pose = tcp_pose @ trans
    camera_pose = back_tcp_pose @ left_camera_to_tcp
    right_camera_pose = camera_pose @ np.linalg.inv(right_camera_to_tcp)
    right_robot.moveToWorldErrorPose(
        right_camera_pose, np.array(jnt_error), vel=3.0, acc=5.0
    )
    time.sleep(0.2)
    color_image, depth_image = camera.get_frame()
    cv2.imwrite(f"../data/good_data/img/{count:05d}.jpg", color_image)
    count += 1
