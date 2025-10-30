import sys

sys.path.append("./")
#print(sys.path)
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
from ldm.models.diffusion.ddpm import LatentDiffusion2
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dino_ddpm_v2 import DinoLatentDiffusion


def transform(img: np.ndarray, device):
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
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
parser.add_argument(
    "--image_only",
    type=bool,
    default=False,
    help="only use image",
)

opt = parser.parse_args()
camera = RealSenseCamera(exposure_time=650)
camera.set_param(camera.fx/640.0*320.0, camera.fy/480.0*240.0,camera.cx/640.0*320.0,camera.cy/480.0*240.0)
if not opt.image_only:
    ip = "172.16.10.33"
    left_base_to_world = np.load("data/vs_examples/extrinsic/left_base_to_world.npy")
    ur_robot = UR(ip, left_base_to_world)
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

config = OmegaConf.load("logs/2025-08-29T20-26-10_charge-m-low-res/configs/2025-08-29T20-26-10-project.yaml")

# input_reference = cv2.imread("data/vs_examples/vs_finetune_charge.jpg")
input_reference = cv2.imread("/mnt/workspace/cyxovo/dataset/2025-08-28-charge-multi-finetune/img/000-012.jpg")


model:DinoLatentDiffusion=instantiate_from_config(config.model)
logging.info("Loading model...")


model.load_state_dict(
    torch.load("logs/2025-08-29T20-26-10_charge-m-low-res/checkpoints/epoch=000618.ckpt")[
        "state_dict"
    ],
    strict=True,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
# static variable
logging.info("Loading tcp pose and camera2tcp...")
tcp_pose = np.load("data/vs_examples/extrinsic/vs_tcp_pose_charge.npy")


camera2tcp = np.load("calibration_data/left_arm/result/camera2tcp.npy")
input_reference = transform(input_reference, device=device)

while True:
    key = input("Move to initial pose?: [Y/n]")
    if key.lower() == "y":
        ur_robot.moveToPose(tcp_pose, vel=0.1, acc=0.1)
    
    if gripper_enable:
        key = input("Open gripper?: [Y/n]")
        if key.lower() == "y":
            gripper.move_and_wait_for_pos(0, 255, 100)
        key = input("Close gripper?: [Y/n]")
        if key.lower() == "y":
            gripper.move_and_wait_for_pos(255, 255, 100)
    key= input(f"Capture image?: [Y/n]") 
    if key.lower() == "n":
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
        pitch_deg = 43
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
        move_distance = 0.10  # 5cm
        target_pos = current_pos + target_dir * move_distance
        # 构建目标位姿（保持当前姿态）
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_rot  # 旋转部分不变
        target_pose[:3, 3] = target_pos  # 更新位置
        ur_robot.moveToWorldPose(target_pose, 0.1, 1)
        break
    if key.lower() == "y":
        count = 0
        if not opt.image_only:
            while count < 30:
                color_image, depth_image = camera.get_frame()
                count += 1
            color_image, depth_image = camera.get_frame()
        
        color_image = cv2.resize(color_image, (320, 240), interpolation=cv2.INTER_AREA)
        conditioning, mask = sam.segment_img(color_image)
        conditioning = transform(conditioning, device=device)
        
        
        print("sam:",conditioning.shape)
        print("mask:",mask.shape)
        print("input_reference:",input_reference.shape)
        
        with torch.no_grad():
            with model.ema_scope():
                c1 = model.cond_stage_model.encode(conditioning)
               
                c2 = model.cond_stage_model.encode(input_reference)
                
                print(c1.shape, c2.shape) 
                c = torch.cat([c1, c2], dim=1)
                c= model.position_enc(c)
               
                shape = (8, 30, 40)
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
                
                print("reference_Image=",reference_image.shape)
                
                _, mask = sam.segment_img(reference_image)
                _, mask_background = sam.segment_img(reference_image)
                print(mask_background.shape)
                mask[mask_background == True] = True
                
    # else:
    #     logging.error("please segment the image to continue")
    #     exit()
    logging.info("Start Servoing...")
    use_roma = True
    
    while True:
        color_image, depth_image = camera.get_frame()

        # 将480x640的color_image缩放为240x320
        color_image = cv2.resize(color_image, (320, 240), interpolation=cv2.INTER_AREA)
        # 对depth_image进行相同的缩放
        depth_image = cv2.resize(depth_image, (320, 240), interpolation=cv2.INTER_AREA)
        
        
        depth_image += 1e-5
        reference_depth = np.zeros_like(depth_image)
        reference_depth[:] = 0.5
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
        if score > 0.8:
            use_roma = True
        vel = linkVelTransform(vel, camera2tcp)
        if use_roma:
            ur_robot.applyTcpVel(vel * 0.2)
        else:
            ur_robot.applyTcpVel(vel * 0.2)
        cv2.imshow("match_img", match_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ur_robot.stop()
            cv2.destroyAllWindows()
            break
        # time.sleep(1 / opt.control_rate)

    
   