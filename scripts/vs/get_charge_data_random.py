import sys
import os

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR
from xlib.device.robotiq.robotiq_gripper import RobotiqGripper
from xlib.device.sensor.camera import RealSenseCamera
from xlib.algo.utils.random import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import time

right_camera_to_tcp = np.load("data/vs_examples/extrinsic/right_camera_to_tcp.npy")
# left_camera_to_tcp = np.load("data/vs_examples/extrinsic/left_camera_to_tcp.npy")
left_camera_to_tcp = np.array(
    [
        [0.99984127, 0.01763296, -0.00255244, -0.00197546],
        [-0.01766463, 0.99975986, -0.01296848, -0.08533614],
        [0.00232316, 0.01301151, 0.99991265, 0.06571628],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
right_base_to_world = np.load("data/vs_examples/extrinsic/right_base_to_world.npy")
# print(left_base_to_world)
# print(right_base_to_world)

enable_gripper = False

left_ip = "172.16.11.33"
right_ip = "172.16.11.68"

# print(left_base_to_world)
left_robot = UR(left_ip, left_base_to_world)
right_robot = UR(right_ip, right_base_to_world)
l515_serial_number = "f1421776"
d435i_serial_number = "233622071298"
camera = RealSenseCamera(exposure_time=700, serial_number=l515_serial_number)

# print(left_robot.world_pose)
# print(right_robot.world_pose)


key = input("activate gripper? (y/n): ")
if key.lower() == "y":
    enable_gripper = True
    left_gripper = RobotiqGripper()
    left_gripper.connect(left_ip, 63352)
    left_gripper.activate()

left_arm_pose = np.eye(4)
left_arm_pose[:3, 3] = np.array([-7.15322882e-02, 4.44952463e-01, 3.34821978e-01])
left_arm_pose[:3, :3] = [
    [1.16551028e-04, -7.05869538e-01, 7.08341853e-01],
    [-9.99999991e-01, -1.32557409e-04, 3.24459005e-05],
    [7.09933881e-05, -7.08341851e-01, -7.05869547e-01],
]
start_right_camera_pose = np.eye(4)
start_right_camera_pose[:3, 3] = np.array([0.3, 0.6, 0.4])
start_right_camera_pose[:3, :3] = R.from_euler(
    "ZYX", [np.pi / 2, 0, -np.pi / 2]
).as_matrix()
jnt_error = [0, 0, 0, 0, 0, 0]
jnt_error[2] = np.pi / 180 * 3.5
left_robot.moveToWorldPose(left_arm_pose, 0.1, 0.1)
right_robot.moveToWorldPose(start_right_camera_pose, vel=0.5, acc=0.5)


if enable_gripper:
    key = input("Close gripper? (y/n): ")
    if key.lower() == "y":
        left_gripper.move_and_wait_for_pos(255, 255, 100)

in_hand_error_pos_lower = [-0.01, -0.02, -0.03]
in_hand_error_pos_upper = [0.01, 0.0, 0.02]
in_hand_error_rot_lower = [-30, -10, -15]
in_hand_error_rot_upper = [-10, 10, 15]

# in_hand_error_pos_lower = [-0.01, -0.03, -0.03]
# in_hand_error_pos_upper = [0.01, 0.0, 0.02]
# in_hand_error_rot_lower = [-25, -10, -20]
# in_hand_error_rot_upper = [-15, 10, 20]


# # change the left
# left_arm_error_pos_lower = np.array([0.0, 0.0, 0.0])
# left_arm_error_pos_upper = np.array([0.0, 0.0, 0.0])
# left_arm_error_rot_lower = np.array([0, 0, 0])
# left_arm_error_rot_upper = np.array([0, 0, 0])


left_arm_error_pos_lower = np.array([-0.003, -0.003, -0.005])
left_arm_error_pos_upper = np.array([0.003, 0.003, 0.005])
left_arm_error_rot_lower = np.array([-5, -5, -5])
left_arm_error_rot_upper = np.array([5, 5, 5])

# left_arm_error_pos_lower = np.array([-0.01, -0.01, -0.02])
# left_arm_error_pos_upper = np.array([0.01, 0.01, 0.0])
# left_arm_error_rot_lower = np.array([-15, -15, -15])
# left_arm_error_rot_upper = np.array([15, 15, 15])


tip_to_tcp = np.eye(4)
tip_to_tcp[2, 3] = 0.18  # 0.275

niuniu_to_tip = np.eye(4)
niuniu_to_tip[2, 3] = 0.18  # 0.015
start_object_pose = left_arm_pose @ tip_to_tcp
start_niuniu_pose = start_object_pose @ niuniu_to_tip

# count big  i small
for i in range(000, 100):
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
    left_robot.moveToWorldPose(cur_left_arm_pose, vel=1.0, acc=1.0, asynchronous=True)
    count = 0
    while count < 30:
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
        tip_pose = cur_object_pose @ (in_hand_error_trans_matrix)
        tcp_pose = tip_pose @ np.linalg.inv(tip_to_tcp)

        # trans = np.eye(4)
        # trans[:3, :3] = R.from_euler("XYZ", [0, 0, np.pi]).as_matrix()
        # back_tcp_pose = tcp_pose @ trans
        camera_pose = tcp_pose @ left_camera_to_tcp
        right_tcp_pose = camera_pose @ np.linalg.inv(right_camera_to_tcp)
        right_robot.moveToWorldPose(
            right_tcp_pose, vel=1.0, acc=1.0, asynchronous=False
        )
        time.sleep(0.2)

        color_image, depth_image = camera.get_frame()

        if color_image is None:
            print("❌ color_image 是 None，图像没有获取到！")
        else:
            print(
                f"✅ color_image shape: {color_image.shape}, dtype: {color_image.dtype}"
            )

        # 确保保存目录存在
        save_dir_image = (
            "/mnt/workspace/cyxovo/dataset/20250912_charge_m/img"
        )

        os.makedirs(save_dir_image, exist_ok=True)
        # 构造完整文件路径
        filename_img = f"{save_dir_image}/{i:03d}-{count:03d}.jpg"

        # 尝试保存图像
        success_img = cv2.imwrite(filename_img, color_image)
        if success_img:
            print(f"✅ 成功保存图片: {filename_img}")
            count += 1
        else:
            print(f"❌ 保存失败: {filename_img}")
            exit()


right_robot.moveToWorldPose(
    start_right_camera_pose,
    vel=1.0,
    acc=1.0,
    asynchronous=False,
)
