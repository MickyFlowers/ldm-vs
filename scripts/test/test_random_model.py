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
from ldm.models.diffusion.dino_ddpm import DinoLatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler


def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


def inverse_transform(image):
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    image = image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    return image


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

root_path = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(
    "logs/2024-12-24T09-55-55_dino_ddpm/configs/2024-12-24T09-55-55-project.yaml"
)
id = "030-003"
input_reference = cv2.imread(
    f"/home/cyx/project/data/good_random_data_single/img/030-004.jpg"
)
model: DinoLatentDiffusion = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load("logs/2024-12-24T09-55-55_dino_ddpm/checkpoints/epoch=000505.ckpt")["state_dict"],
    strict=True,
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
# static variable
logging.info("Loading tcp pose and camera2tcp...")
tcp_pose = np.load("data/vs_examples/extrinsic/vs_tcp_pose.npy")
camera2tcp = np.load("data/vs_examples/extrinsic/left_camera_to_tcp.npy")
input_reference = transform(input_reference, device=device)
# conditioning = cv2.imread(
#     f"/home/cyx/project/data/good_random_data_single/seg/{id}.jpg"
# )
# conditioning, mask = sam.segment_img(color_image)
conditioning = cv2.imread(
    f"/home/cyx/project/data/good_random_data_single/seg/{id}.jpg"
)
conditioning = transform(conditioning, device=device)
shape = (8, 60, 80)
with torch.no_grad():
    with model.ema_scope():
        c1 = model.cond_stage_model.encode(conditioning)
        c2 = model.cond_stage_model.encode(input_reference)
        c = torch.cat([c1, c2], dim=1)
        sample, _ = sampler.sample(
            S=opt.steps,
            conditioning=c,
            batch_size=c.shape[0],
            shape=shape,
            verbose=False,
            eta=0,
            # x_T=torch.zeros(c1.shape, device=device),
        )
        x_sample = model.first_stage_model.decode(sample)
        reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        reference_image = reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
        reference_image = reference_image.astype(np.uint8)
        cv2.imshow("reference_image", reference_image)
        cv2.imshow("input_reference", inverse_transform(input_reference))
        cv2.waitKey(0)
