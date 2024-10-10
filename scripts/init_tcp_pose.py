import sys

sys.path.append("./")
from xlib.device.manipulator.ur_robot import UR
import numpy as np

ip = "10.51.33.233"
robot = UR(ip)
tcp_pose = robot.tcp_pose
np.save("data/vs_examples/extrinsic/vs_tcp_pose.npy", tcp_pose)
