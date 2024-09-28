import sys

sys.path.append("./")
from xlib.algo.ibvs import IBVS
from xlib.algo.kp_matcher import RomaMatchAlgo
from xlib.device.manipulator.ur_robot import UR
from xlib.device.robotiq import robotiq_gripper
from xlib.device.sensor.camera import RealSenseCamera
from xlib.sam.sam_gui import SAM
from xlib.algo.transforms import linkVelTransform
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
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler


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
    default=500,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--control_rate",
    type=int,
    default=100,
    help="number of ddim sampling steps",
)
opt = parser.parse_args()

camera = RealSenseCamera(exposure_time=400)
ip = "192.168.1.102"
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
# ibvs_controller = IBVS(camera)
ibvs_controller = IBVS(camera, kp_extractor=RomaMatchAlgo)
sam = SAM()
root_path = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(
    "logs/2024-09-21T10-17-38_vs/configs/2024-09-21T10-17-38-project.yaml"
)
model: LatentDiffusion = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load("logs/2024-09-21T10-17-38_vs/checkpoints/last.ckpt")["state_dict"],
    strict=True,
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
# static variable
logging.info("Loading tcp pose and camera2tcp...")
tcp_pose = np.load("data/vs_examples/vs_tcp_pose.npy")
camera2tcp = np.load("data/vs_examples/camera2tcp.npy")

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

    key = input("Capture image?: [Y/n]")
    if key.lower() == "y":
        color_image, depth_image = camera.get_frame()
        conditioning, mask = sam.segment_img(color_image)
        conditioning = transform(conditioning, device=device)
        with torch.no_grad():
            with model.ema_scope():
                c = model.cond_stage_model.encode(conditioning).mode()
                sample, _ = sampler.sample(
                    S=opt.steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=c.shape[1:],
                    verbose=False,
                )
                x_sample = model.first_stage_model.decode(sample)
                reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                reference_image = (
                    reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                )
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
                reference_image = reference_image.astype(np.uint8)
    else:
        logging.error("please segment the image to continue")
        exit()
    logging.info("Start Servoing...")
    while True:
        color_image, depth_image = camera.get_frame()
        reference_depth = np.zeros_like(depth_image)
        reference_depth[:] = 0.3
        vel, score, match_img = ibvs_controller.cal_vel_from_img(
            reference_image, color_image, reference_depth, depth_image, mask, True
        )
        vel = linkVelTransform(vel, camera2tcp)
        ur_robot.applyTcpVel(vel * 0.2)
        cv2.imshow("match_img", match_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ur_robot.stop()
            cv2.destroyAllWindows()
            break
        time.sleep(1 / opt.control_rate)
