import sys

sys.path.append("./")
from xlib.algo.vs.vs_controller.ibvs import IBVS, CNSV2

# CNS #
from simple_client import SimpleClient

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

camera = RealSenseCamera(exposure_time=500, serial_number="f1421776")
ip = "172.16.11.33"
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
cns_controller = CNSV2(camera=camera)
# CNS #
# controller_dif = SimpleClient("CNSv2_m0117")

sam = SAM()
root_path = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(
    "/mnt/workspace/cyxovo/model/ldm/2024-12-25T14-08-32_dino_ddpm_single/configs/2024-12-25T14-08-32-project.yaml"
)
# config = OmegaConf.load(
#     "logs/2025-04-21-dino_ddpm/2025-04-20T11-03-54-project.yaml"
# )  #修改配置
id = "031-001"
model: DinoLatentDiffusionSingle = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load(
        "/mnt/workspace/cyxovo/model/ldm/2024-12-25T14-08-32_dino_ddpm_single/checkpoints/epoch=000179.ckpt"
    )["state_dict"],
    strict=True,
)
# model.load_state_dict(
#     torch.load(
#         "logs/2024-12-25T14-08-32_dino_ddpm_single/checkpoints/epoch=000179.ckpt"
#     )["state_dict"],
#     strict=True,
# )
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
# static variable
logging.info("Loading tcp pose and camera2tcp...")
tcp_pose = np.load("data/vs_examples/extrinsic/vs_tcp_pose.npy")
camera2tcp = np.load("calibration_data/left_arm/result/camera2tcp.npy")
# camera2tcp = np.load("data/vs_examples/extrinsic/left_camera_to_tcp.npy")
while True:
    key = input("Move to initial pose?: [Y/n]")
    if key.lower() == "y":
        ur_robot.moveToPose(tcp_pose, vel=0.1, acc=0.1)

    gripper_enable = True
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
        conditioning, mask, _, _ = sam.segment_img(color_image)

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
                reference_image = cv2.imread("data/vs_examples/vs_finetune.jpg")
                _, mask_background, _, _ = sam.segment_img(reference_image)

                _, mask, _, _ = sam.segment_img(reference_image)
                mask[mask_background == True] = True

                mask_end_float = mask.astype(np.float32)
                cv2.imshow("maskend", mask_end_float)
                cv2.waitKey(0)

    else:
        logging.error("please segment the image to continue")
        exit()
    logging.info("Start Servoing...")
    use_roma = False
    use_cns = True

    init_cns = True
    while True:
        color_image, depth_image = camera.get_frame()
        # if use_cns:
        #     if init_cns:
        #         tar_bgr = reference_image
        #         tar_depth = depth_image
        #         depth_hint = np.array(0.2)
        #         tar_rgb = np.ascontiguousarray(tar_bgr[:, :, [2, 1, 0]])
        #         tar_mask = mask
        #         cam_intr = camera.get_color_intr2()
        #         xx, yy = np.meshgrid(
        #             np.arange(tar_bgr.shape[1]), np.arange(tar_bgr.shape[0]), indexing="xy"
        #         )
        #         grid = np.stack([xx, yy], axis=-1)  # (H, W, 2)
        #         norm_xy = cam_intr.pixel_to_norm_camera_plane(grid)
        #         tar_rgb = tar_rgb.copy()
        #         tar_rgb[tar_mask] = 0
        #         tar_bgr = np.ascontiguousarray(tar_rgb[:, :, [2, 1, 0]])

        #         controller_dif.set_target(tar_bgr, cam_intr.K, depth_hint, None)
        #         init_cns=False

        # conditioning = np.zeros_like(color_image)
        # conditioning[mask] = color_image[mask]
        # conditioning = transform(conditioning, device=device)
        # with torch.no_grad():
        #     with model.ema_scope():
        #         c = model.cond_stage_model.encode(conditioning).mode()
        #         sample, _ = sampler.sample(
        #             S=opt.steps,
        #             conditioning=c,
        #             batch_size=c.shape[0],
        #             shape=c.shape[1:],
        #             verbose=False,
        #             x_T=torch.zeros(c.shape, device=device),
        #         )05, -2.4658421e-03], dtype=float3
        #         reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
        #         reference_image = reference_image.astype(np.uint8)
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
        elif use_cns:
            cns_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
            )
            _, vel, score, match_img = cns_controller.calc_vel(
                depth_hint=0.2, mask=mask
            )
        else:
            ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = ibvs_controller.calc_vel(mask=mask)
        logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
        # if score > 0.75:
        #     use_roma = True
        vel = linkVelTransform(vel, camera2tcp)
        if use_roma:
            ur_robot.applyTcpVel(vel * 0.15)
        else:
            ur_robot.applyTcpVel(vel)
        cv2.imshow("match_img", match_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ur_robot.stop()
            cv2.destroyAllWindows()
            break
        # time.sleep(1 / opt.control_rate)
