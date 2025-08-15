from xlib.device.manipulator.ur_robot import UR
import numpy as np
from scipy.spatial.transform import Rotation as R
ip = "172.16.10.33"
ur_robot = UR(ip)


# 获取当前TCP位姿（基坐标系下）
cur_pose = ur_robot.tcp_pose
current_pos = cur_pose[:3, 3]
current_rot = cur_pose[:3, :3]

# 设置向下倾斜角度（更平缓）
angle_deg = 30  # 或者用 np.random.uniform(30, 45)
angle_rad = np.deg2rad(angle_deg)
dir_tcp = np.array([0, -np.sin(angle_rad), np.cos(angle_rad)])
# 转换到基坐标系
dir_base = current_rot @ dir_tcp

# 计算目标位置（基坐标系下）
target_pos = current_pos + dir_base * 0.05

# 保持旋转不变，构造目标位姿
target_pose = np.eye(4)
target_pose[:3, :3] = current_rot  # 旋转部分不变
target_pose[:3, 3] = target_pos   # 更新位置

# 转换为UR的6维向量 [x,y,z,rx,ry,rz]
target_pose_vec = np.zeros(6)
target_pose_vec[:3] = target_pos
target_pose_vec[3:] = R.from_matrix(current_rot).as_rotvec()
ur_robot.rtde_c.moveL(target_pose_vec.tolist(), 0.1, 1)
