import sys

sys.path.append("./")
from xlib.algo.vs.vs_controller.ibvs import IBVS
from xlib.algo.vs.kp_matcher import RomaMatchAlgo, KpMatchAlgo
from xlib.device.manipulator.ur_robot import UR
from xlib.device.robotiq import robotiq_gripper
from xlib.device.sensor.camera import RealSenseCamera
from xlib.sam.sam_gui import SAM
from xlib.algo.utils.transforms import linkVelTransform
import xlib.log
import logging
import numpy as np
import cv2
import os
import argparse
import torch
import time
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dino_ddpm import DinoLatentDiffusionSingle
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# camera = RealSenseCamera(exposure_time=800,serail_number='f1421776')
# ibvs_controller = IBVS(camera, kp_algo=KpMatchAlgo, match_threshold=0.4)
# roma_ibvs_controller = IBVS(camera, kp_algo=RomaMatchAlgo)
# sam = SAM()
# ip = "172.16.10.33"
# ur_robot = UR(ip)
# camera2tcp = np.load("calibration_data/left_arm/result/camera2tcp.npy")
# reference_image = cv2.imread("data/vs_examples/vs_charge.jpg")

# _, mask = sam.segment_img(reference_image)
# _, mask_background = sam.segment_img(reference_image)

# mask[mask_background == True] = True
# use_roma = True
# while True:
#     color_image, depth_image = camera.get_frame()
#     depth_image += 1e-5
#     reference_depth = np.zeros_like(depth_image)
#     reference_depth[:] = 0.4
#     if use_roma:
#         roma_ibvs_controller.update(
#             cur_img=color_image,
#             tar_img=reference_image,
#             cur_depth=depth_image,
#             tar_depth=reference_depth,
#         )
#         _, vel, score, match_img = roma_ibvs_controller.calc_vel(mask=mask)
#     else:
#         ibvs_controller.update(
#             cur_img=color_image,
#             tar_img=reference_image,
#             cur_depth=depth_image,
#             tar_depth=reference_depth,
#         )
#         _, vel, score, match_img = ibvs_controller.calc_vel(mask=mask)
#     logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
#     if score > 0.75:
#         use_roma = True
#     vel = linkVelTransform(vel, camera2tcp)
#     if use_roma:
#         ur_robot.applyTcpVel(vel * 0.15)
#     else:
#         ur_robot.applyTcpVel(vel * 0.15)
#     cv2.imshow("match_img", match_img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         ur_robot.stop()
#         cv2.destroyAllWindows()
#         break


def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--control_rate",
    type=int,
    default=100,
    help="number of ddim sampling steps",
)
opt = parser.parse_args()

camera = RealSenseCamera(exposure_time=800,serail_number='f1421776')
ip = "172.16.10.33"
ur_robot = UR(ip)
# activate gripper
gripper = robotiq_gripper.RobotiqGripper()
gripper.connect(ip, 63352)
gripper_enable = False
key = input("Activate gripper?: [Y/n]")
if key.lower() == "y":
    gripper_enable = True

if gripper_enable:
    gripper.activate()
use_roma = True
ibvs_controller = IBVS(camera, kp_algo=KpMatchAlgo, match_threshold=0.4)
roma_ibvs_controller = IBVS(camera, kp_algo=RomaMatchAlgo)
sam = SAM()
root_path = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(
    "logs/2025-07-18T02-13-09_dino_ddpm_charge/configs/2025-07-18T02-13-09-project.yaml"
)
id = "031-001"
model: DinoLatentDiffusionSingle = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load(
        "logs/2025-07-18T02-13-09_dino_ddpm_charge/checkpoints/epoch=000218.ckpt"
    )["state_dict"],
    strict=True,
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
# static variable
logging.info("Loading tcp pose and camera2tcp...")
tcp_pose = np.load("data/vs_examples/extrinsic/vs_tcp_pose.npy")
camera2tcp = np.load("calibration_data/left_arm/result/camera2tcp.npy")
# camera2tcp = np.load("data/vs_examples/extrinsic/left_camera_to_tcp.npy")
while True:
    key = input("Move in charge?: [Y/n]")
    if key.lower() == "y":
        # 获取当前TCP位姿（基坐标系下）
        cur_pose = ur_robot.tcp_pose
        current_pos = cur_pose[:3, 3]
        current_rot = cur_pose[:3, :3]

        # 设置向下倾斜角度（更平缓）
        angle_deg = 35  # 或者用 np.random.uniform(30, 45)
        angle_rad = np.deg2rad(angle_deg)
        dir_tcp = np.array([0, -np.sin(angle_rad), np.cos(angle_rad)])
        # 转换到基坐标系
        dir_base = current_rot @ dir_tcp

        # 计算目标位置（基坐标系下）
        target_pos = current_pos + dir_base * 0.08

        # 保持旋转不变，构造目标位姿
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_rot  # 旋转部分不变
        target_pose[:3, 3] = target_pos   # 更新位置

        # 转换为UR的6维向量 [x,y,z,rx,ry,rz]
        target_pose_vec = np.zeros(6)
        target_pose_vec[:3] = target_pos
        target_pose_vec[3:] = R.from_matrix(current_rot).as_rotvec()
        ur_robot.rtde_c.moveL(target_pose_vec.tolist(), 0.1, 1)
        gripper.move_and_wait_for_pos(0, 255, 100)
        break
    key = input("Move to initial pose?: [Y/n]")
    if key.lower() == "y":
        ur_robot.moveToPose(tcp_pose, vel=0.1, acc=0.1)
        
    gripper_enable=True
    if gripper_enable:
        key = input("Open gripper?: [Y/n]")
        if key.lower() == "y":
            gripper.move_and_wait_for_pos(0, 255, 100)
        key = input("Close gripper?: [Y/n]")
        if key.lower() == "y":
            gripper.move_and_wait_for_pos(255, 255, 100)

    key = input("Capture image?: [Y/n]")
    if key.lower() == "y":
        count = 0
        while count < 30:
            color_image, depth_image = camera.get_frame()
            count += 1
        color_image, depth_image = camera.get_frame()
        conditioning, mask = sam.segment_img(color_image)

        # mask_float = mask.astype(np.float32)  # True -> 1.0, False -> 0.0
        # cv2.imshow("mask", mask_float)
        # cv2.waitKey(0)

        conditioning = transform(conditioning, device=device)
        shape = (8, 60, 80)
        with torch.no_grad():
            with model.ema_scope():
                c = model.cond_stage_model.encode(conditioning)
                print(c.shape)
                sample, _ = sampler.sample(
                    S=opt.steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=shape,
                    verbose=False,
                    eta=0,
                    x_T=torch.zeros((c.shape[0], *shape), device=device),
                )
                x_sample = model.first_stage_model.decode(sample)
                reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                reference_image = (
                    reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                )
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
                reference_image = reference_image.astype(np.uint8)

                _, mask_background = sam.segment_img(reference_image)

                _, mask = sam.segment_img(reference_image)
                mask[mask_background == True] = True

                # mask_end_float=mask.astype(np.float32)
                # cv2.imshow("maskend",mask_end_float)
                # cv2.waitKey(0)

    else:
        logging.error("please segment the image to continue")
        exit()
    logging.info("Start Servoing...")
    use_roma = True
    while True:
        color_image, depth_image = camera.get_frame()
        depth_image += 1e-5
        reference_depth = np.zeros_like(depth_image)
        reference_depth[:] = 0.4
        if use_roma:
            roma_ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = roma_ibvs_controller.calc_vel(mask=mask)
        else:
            ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = ibvs_controller.calc_vel(mask=mask)
        logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
        if score > 0.75:
            use_roma = True
        vel = linkVelTransform(vel, camera2tcp)
        if use_roma:
            ur_robot.applyTcpVel(vel * 0.15)
        else:
            ur_robot.applyTcpVel(vel * 0.15)
        cv2.imshow("match_img", match_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ur_robot.stop()
            cv2.destroyAllWindows()
            break
        # time.sleep(1 / opt.control_rate)
