import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR

left_ip = "172.16.11.33"

# 加载从机械臂基座到世界坐标系的变换矩阵
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")

# 创建UR机械臂实例
left_robot = UR(left_ip, left_base_to_world)

# 获取当前机械臂的位姿（4x4齐次变换矩阵）
current_pose = left_robot.tcp_pose

# 提取位置向量（XYZ坐标）
tcp_position = current_pose[:3, 3]

# 提取旋转矩阵
rotation_matrix = current_pose[:3, :3]

# 打印结果
print("\n可以直接复制到代码的形式:")
print(
    f"left_arm_pose[:3, 3] = np.array([{tcp_position[0]:.8e}, {tcp_position[1]:.8e}, {tcp_position[2]:.8e}])"
)
print(rotation_matrix)
np.save("data/vs_examples/extrinsic/vs_tcp_pose_charge.npy", current_pose)

# print(current_pose)
