import sys

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR
import numpy as np
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
right_base_to_world = np.load("data/vs_examples/extrinsic/right_base_to_world.npy")
ip = "10.51.33.233"
robot = UR(ip, left_base_to_world)
tcp_pose = robot.world_pose
print(tcp_pose)
# np.save("data/vs_examples/extrinsic/vs_tcp_pose.npy", tcp_pose)
