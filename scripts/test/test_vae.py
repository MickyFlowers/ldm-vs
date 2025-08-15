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
from ldm.models.autoencoder import AutoencoderKL

    
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
root_path = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(
    "logs/2024-12-14T10-17-42_vs_random_cond_simple/configs/2024-12-14T10-17-42-project.yaml"
)
input_image = cv2.imread("../data/good_random_data_single/seg/000-000.jpg")
input_tensor = transform(input_image, device)
model: AutoencoderKL = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load(
        "logs/2024-12-14T10-17-42_vs_random_cond_simple/checkpoints/epoch=000011.ckpt"
    )["state_dict"],
    strict=True,
)
model = model.to(device)
with torch.no_grad():
    rec, _ = model(input_tensor)
    rec_image = inverse_transform(rec)
    error = cv2.absdiff(input_image, rec_image)
    cv2.imshow("input", input_image)
    cv2.imshow("rec", rec_image)
    cv2.imshow("error", error)
    cv2.waitKey(0)
