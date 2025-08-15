import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import cv2
import numpy as np


class InferenceWrapper:
    def __init__(self, config_path, ckpt_path, device=None):
        config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(
            torch.load("models/ldm/ddpm_vs/last.ckpt")["state_dict"],
            strict=True,
        )
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)

    @torch.no_grad()
    def inference(
        self,
        reference: np.ndarray,
        segmentation: np.ndarray,
        latent_shape=(8, 60, 80),
        steps: int = 50,
        position_enc: bool = True,
    ):
        reference = to_tensor(reference, self.device)
        segmentation = to_tensor(segmentation, self.device)
        with self.model.ema_scope():
            c1 = self.model.cond_stage_model.encode(reference)
            c2 = self.model.cond_stage_model.encode(segmentation)
            c = torch.cat([c1, c2], dim=1)
            if position_enc:
                c = self.model.position_enc(c)
            sample, _ = self.sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=latent_shape,
                verbose=False,
                eta=0,
                x_T=torch.zeros((c.shape[0], *latent_shape), device=self.device),
            )
            x_sample = self.model.first_stage_model.decode(sample)
            x_sample = to_numpy(x_sample)
            return x_sample


def to_tensor(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    tensor = img * 2.0 - 1.0
    tensor = tensor.to(device)
    return tensor


def to_numpy(img_tensor: torch.Tensor):
    image = torch.clamp((img_tensor + 1.0) / 2.0, min=0.0, max=1.0)
    image = image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    return image
