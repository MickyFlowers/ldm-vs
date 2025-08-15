# enrich the distribution of the data

import sys

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR
from xlib.device.robotiq.robotiq_gripper import RobotiqGripper
from xlib.device.sensor.camera import RealSenseCamera
from xlib.algo.utils.random import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import time

root_path = "../data/good_data_single"
right_camera_to_tcp = np.load("data/vs_examples/extrinsic/right_camera_to_tcp.npy")
left_camera_to_tcp = np.load("data/vs_examples/extrinsic/left_camera_to_tcp.npy")
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
right_base_to_world = np.load("data/vs_examples/extrinsic/right_base_to_world.npy")
# print(left_base_to_world)
# print(right_base_to_world)

enable_gripper = False

left_ip = "172.16.11.33"
right_ip = "172.16.11.111"

# print(left_base_to_world)
left_robot = UR(left_ip, left_base_to_world)
right_robot = UR(right_ip, right_base_to_world)
l515_serial_number = "f1421776"
d435i_serial_number = "233622071298"
camera = RealSenseCamera(exposure_time=500, serail_number=l515_serial_number)

# print(left_robot.world_pose)
# print(right_robot.world_pose)


key = input("activate gripper? (y/n): ")
if key.lower() == "y":
    enable_gripper = True
    left_gripper = RobotiqGripper()
    left_gripper.connect(left_ip, 63352)
    left_gripper.activate()

left_arm_pose = np.eye(4)
left_arm_pose[:3, 3] = np.array([-1.51300577e-02, 5.15169458e-01, 4.14093506e-01])
left_arm_pose[:3, :3] = R.from_euler("XYZ", [np.pi, 0, -np.pi / 2]).as_matrix()
start_right_camera_pose = np.eye(4)
start_right_camera_pose[:3, 3] = np.array([0.2, 0.5, 0.3])
start_right_camera_pose[:3, :3] = R.from_euler(
    "ZYX", [np.pi / 2, 0, -np.pi / 2]
).as_matrix()
jnt_error = [0, 0, 0, 0, 0, 0]
jnt_error[2] = np.pi / 180 * 3.5
right_robot.moveToWorldErrorPose(
    start_right_camera_pose, np.array(jnt_error), vel=1.0, acc=1.0
)
left_robot.moveToWorldPose(left_arm_pose, 0.1, 0.1)
if enable_gripper:
    key = input("Close gripper? (y/n): ")
    if key.lower() == "y":
        left_gripper.move_and_wait_for_pos(255, 255, 100)

# in_hand_error_pos_lower = [0.0, -0.02, -0.03]
# in_hand_error_pos_upper = [0.0, -0.02, -0.03]
# in_hand_error_rot_lower = [-20, 0.0, 0.0]
# in_hand_error_rot_upper = [-20, 0.0, 0.0]


in_hand_error_pos_lower = [-0.01, -0.02, -0.03]
in_hand_error_pos_upper = [0.01, 0.0, 0.02]
in_hand_error_rot_lower = [-45, -10, -20]
in_hand_error_rot_upper = [-20, 10, 20]
# left_arm_error_pos_lower = np.array([-0.03, -0.06, -0.03])
# left_arm_error_pos_upper = np.array([0.03, 0.06, 0.0])
# left_arm_error_rot_lower = np.array([-15, -15, -15])
# left_arm_error_rot_upper = np.array([15, 15, 15])

left_arm_error_pos_lower = np.array([0.0, 0.0, 0.0])
left_arm_error_pos_upper = np.array([0.0, 0.0, 0.0])
left_arm_error_rot_lower = np.array([0, 0, 0])
left_arm_error_rot_upper = np.array([0, 0, 0])

tip_to_tcp = np.eye(4)
tip_to_tcp[2, 3] = 0.16

niuniu_to_tip = np.eye(4)
niuniu_to_tip[2, 3] = 0.15
start_object_pose = left_arm_pose @ tip_to_tcp
start_niuniu_pose = start_object_pose @ niuniu_to_tip
for i in range(5):
    color_image, depth_image = camera.get_frame()

for i in range(2853, 10000):
    left_arm_error_pos = np.random.uniform(
        left_arm_error_pos_lower, left_arm_error_pos_upper
    )
    left_arm_error_rot = np.random.uniform(
        left_arm_error_rot_lower, left_arm_error_rot_upper
    )
    left_arm_error_trans_matrix = np.eye(4)
    left_arm_error_trans_matrix[:3, :3] = R.from_euler(
        "XYZ", left_arm_error_rot, degrees=True
    ).as_matrix()
    left_arm_error_trans_matrix[:3, 3] = left_arm_error_pos
    cur_niuniu_pose = start_niuniu_pose @ left_arm_error_trans_matrix
    cur_object_pose = cur_niuniu_pose @ np.linalg.inv(niuniu_to_tip)
    cur_left_arm_pose = cur_object_pose @ np.linalg.inv(tip_to_tcp)
    left_robot.moveToWorldPose(cur_left_arm_pose, vel=1.0, acc=1.0, asynchronous=False)
    count = 0
    # while count < 2:
    in_hand_error_pos = np.random.uniform(
        in_hand_error_pos_lower, in_hand_error_pos_upper
    )
    in_hand_error_rot = np.random.uniform(
        in_hand_error_rot_lower, in_hand_error_rot_upper
    )
    in_hand_error_trans_matrix = np.eye(4)
    in_hand_error_trans_matrix[:3, :3] = R.from_euler(
        "XYZ", in_hand_error_rot, degrees=True
    ).as_matrix()
    in_hand_error_trans_matrix[:3, 3] = in_hand_error_pos
    tip_pose = cur_object_pose @ np.linalg.inv(in_hand_error_trans_matrix)
    tcp_pose = tip_pose @ np.linalg.inv(tip_to_tcp)

    trans = np.eye(4)
    trans[:3, :3] = R.from_euler("XYZ", [0, 0, np.pi]).as_matrix()
    back_tcp_pose = tcp_pose @ trans
    camera_pose = back_tcp_pose @ left_camera_to_tcp
    right_camera_pose = camera_pose @ np.linalg.inv(right_camera_to_tcp)
    right_robot.moveToWorldPose(right_camera_pose, vel=1.0, acc=1.0, asynchronous=False)
    success = right_robot.moveToWorldErrorPose(
        right_camera_pose, np.array(jnt_error), vel=1.0, acc=1.0, asynchronous=False
    )
    time.sleep(0.2)
    if success:
        color_image, depth_image = camera.get_frame()
        cv2.imwrite(
            f"{root_path}/img/{i:05d}.jpg",
            color_image,
        )
        count += 1
    else:
        print("Failed")

    # right_robot.moveToWorldErrorPose(
    #     start_right_camera_pose,
    #     np.array(jnt_error),
    #     vel=2.0,
    #     acc=3.0,
    #     asynchronous=True,
    # )
