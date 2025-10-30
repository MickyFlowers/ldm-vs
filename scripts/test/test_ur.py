from xlib.device.manipulator.ur_robot import UR
import numpy as np
from scipy.spatial.transform import Rotation as R
from xlib.device.robotiq import robotiq_gripper

ip = "172.16.11.33"
left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
ur_robot = UR(ip, left_base_to_world)


world_pose = ur_robot.world_pose

world_pose[2,3]+=0.1

ur_robot.moveToWorldPose(world_pose)
