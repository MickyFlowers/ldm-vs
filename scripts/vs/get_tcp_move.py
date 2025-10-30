from xlib.device.manipulator.ur_robot import UR
import numpy as np
from scipy.spatial.transform import Rotation as R
from xlib.device.robotiq import robotiq_gripper

ip = "172.16.11.33"
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
ur_robot = UR(ip, left_base_to_world)


# expect_pose = np.load("charge_world_expect.npy")
# ur_robot.moveToWorldPose(expect_pose, 0.1, 0.1)


# 获取当前TCP位姿（世界坐标系下）
cur_pose = ur_robot.world_pose
current_pos = cur_pose[:3, 3]  # TCP当前位置 [x, y, z]
current_rot = cur_pose[:3, :3]  # TCP当前姿态（旋转矩阵）

# 获取TCP的Z轴方向（世界坐标系下的当前朝向）
tcp_z_axis = current_rot[:, 2]  # 旋转矩阵的第3列

# 计算当前朝向的水平分量（在XY平面的投影）
horizontal_dir = np.array([tcp_z_axis[0], tcp_z_axis[1], 0])
if np.linalg.norm(horizontal_dir) > 0:
    horizontal_dir = horizontal_dir / np.linalg.norm(horizontal_dir)

# 定义俯仰角（45度斜向下）越小越缓
pitch_deg = 45
pitch_rad = np.deg2rad(pitch_deg)

# 计算目标方向向量（保持水平方向）构造一个单位向量，并且与世界坐标系z轴夹角为--45度
target_dir = np.array(
    [
        horizontal_dir[0] * np.cos(pitch_rad),
        horizontal_dir[1] * np.cos(pitch_rad),
        -np.sin(pitch_rad),  # 负号表示向下
    ]
)

# 移动距离
move_distance = -0.025  # 5cm
# move_distance = 0.01  # 5cm
target_pos = current_pos + target_dir * move_distance
# 构建目标位姿（保持当前姿态）
target_pose = np.eye(4)
target_pose[:3, :3] = current_rot  # 旋转部分不变
target_pose[:3, 3] = target_pos  # 更新位置
ur_robot.moveToWorldPose(target_pose, 0.1, 1)


# gripper = robotiq_gripper.RobotiqGripper()
# gripper.connect(ip, 63352)
# gripper.move_and_wait_for_pos(0, 255, 100)
