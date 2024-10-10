import sys

sys.path.append("./")
from xlib.algo.calibrator import EyeHandCalibrator
from xlib.device.sensor.camera import RealSenseCamera
from xlib.device.manipulator.ur_robot import UR
import xlib.log
import numpy as np
from scipy.spatial.transform import Rotation as R

camera = RealSenseCamera()
ur_robot = UR(ip="10.51.33.233")
cali = EyeHandCalibrator(
    camera,
    ur_robot,
)

cali.setAruco()
cali.sampleImages("./calibration_data/right_arm")
cali.calibrate("eye-in-hand", "./calibration_data/right_arm")
