import sys

sys.path.append("./")
from xlib.algo.calibrator import EyeHandCalibrator
from xlib.device.sensor.camera import RealSenseCamera
from xlib.device.manipulator.ur_robot import UR
import xlib.log
import numpy as np
from scipy.spatial.transform import Rotation as R

camera = RealSenseCamera()
# exit()
ur_robot = UR(ip="172.16.8.33")
cali = EyeHandCalibrator(
    camera,
    ur_robot,
)

cali.setAruco()
cali.sampleImages("./calibration_data/right_arm")
cali.calibrate("eye-in-hand", "./calibration_data/right_arm")
