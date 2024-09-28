import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
import cv2
import torch
import numpy as np


def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()
    config = OmegaConf.load(
        "logs/2024-09-21T10-17-38_vs/configs/2024-09-21T10-17-38-project.yaml"
    )
    model: LatentDiffusion = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load("logs/2024-09-21T10-17-38_vs/checkpoints/last.ckpt")[
            "state_dict"
        ],
        strict=True,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    conditioning = cv2.imread("data/vs_examples/00001-seg.jpg")
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
            predicted_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_image = (
                predicted_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
            )
            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("data/vs_examples/00001-pre.jpg", predicted_image)
