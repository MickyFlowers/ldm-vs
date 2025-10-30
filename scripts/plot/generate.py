import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from ldm.models.diffusion.dino_ddpm_v2 import DinoLatentDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dino_ddpm_v2 import DinoLatentDiffusion
from xlib.sam.sam_gui import SAM
import argparse


def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


def main(args):
    sam = SAM()
    config = OmegaConf.load(args.config)
    model: DinoLatentDiffusion = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(args.ckpt_path)["state_dict"], strict=True)
    model.eval()
    model.to("cuda")
    sampler = DDIMSampler(model)
    input_reference = cv2.imread(args.input_reference_path)
    input_reference = cv2.resize(input_reference, (0, 0), fx=args.scale, fy=args.scale)
    input_reference = transform(input_reference, "cuda")
    color_image = cv2.imread(args.input)
    conditioning, mask, _, _ = sam.segment_img(color_image)
    cv2.imwrite("temp/conditioning.jpg", conditioning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scale",
    type=float,
    default=0.5,
    help="image size scale",
)
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--config",
    type=str,
    default="logs/2025-08-15T17-55-08_screwdriver-m/configs/2025-08-15T17-55-08-project.yaml",
    help="config file",
)
parser.add_argument(
    "--input_reference_path",
    type=str,
    default="data/vs_examples/vs_finetune.jpg",
    help="input image",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="logs/2025-08-15T17-55-08_screwdriver-m/checkpoints/epoch=000063.ckpt",
    help="ckpt path",
)
parser.add_argument(
    "--input",
    type=str,
    default="/home/cyx/project/latent-diffusion/temp/video/backup/10101309/first_image/000.jpg",
)
opt = parser.parse_args()
main(opt)
