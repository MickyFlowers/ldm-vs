import torchvision
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageFilter
import os
import random
import torch
import piq
import time
from tqdm import tqdm
import torch.nn.functional as F
from xlib.algo.utils.metric import *
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion2
from ldm.models.diffusion.ddim import DDIMSampler
import argparse
import cv2

def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def enhance_image(image):

    brightness_factor = random.uniform(0.5, 2.0)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    contrast_factor = random.uniform(0.5, 2.0)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    color_factor = random.uniform(0.5, 2.0)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    sharpness_factor = random.uniform(0.5, 2.0)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    return image


def pil_loader(path):
    return Image.open(path).convert("RGB")


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str_, encoding="utf-8")]
    else:
        images = []
        assert os.path.isdir(dir), "%s is not a valid directory" % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


class VsRandomDataset(data.Dataset):
    def __init__(self, data_root, data_flist=None, img_size=[480, 640], **kwargs):
        super().__init__()
        self.data_root = data_root
        self.data_flist = data_flist
        self.img_size = img_size
        self.processer = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.img_size[0], self.img_size[1])),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        self.loader = pil_loader
        data_flist = os.path.join(self.data_root, self.data_flist)
        self.flist = make_dataset(data_flist)

    def __getitem__(self, index):
        ret = {}
        pose_count = int(self.flist[index]) // 2500
        peg_count = (int(self.flist[index]) % 2500) // 50
        ref_count = (int(self.flist[index]) % 2500) % 50

        ref_img_file_name = os.path.join(
            self.data_root, "img", f"{pose_count:03d}-{ref_count:03d}.jpg"
        )
        peg_count_file_name = os.path.join(
            self.data_root, "seg", f"{pose_count:03d}-{peg_count:03d}.jpg"
        )
        gt_count_file_name = os.path.join(
            self.data_root, "img", f"{pose_count:03d}-{peg_count:03d}.jpg"
        )
        ref_img = self.loader(ref_img_file_name)
        peg_img = self.loader(peg_count_file_name)
        gt_image = self.loader(gt_count_file_name)
        ref_img = self.processer(ref_img)
        peg_img = self.processer(peg_img)
        gt_image = self.processer(gt_image)
        ret["image"] = gt_image
        ret["segmentation"] = peg_img
        ret["reference"] = ref_img
        return ret

    def __len__(self):
        return len(self.flist)


def inverse_transform(image):
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    image = image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    return image


if __name__ == "__main__":
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
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    data_root = "/mnt/workspace/cyxovo/dataset/good_random_data_single"
    train_data_flist = "train_vs.flist"
    test_data_flist = "val_vs.flist"
    train_dataset = VsRandomDataset(data_root, train_data_flist)
    test_dataset = VsRandomDataset(data_root, test_data_flist)
    # model
    config = OmegaConf.load(
        "logs/2024-12-24T09-55-55_dino_ddpm/configs/2024-12-24T09-55-55-project.yaml"
    )
    model: LatentDiffusion2 = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load("logs/2024-12-24T09-55-55_dino_ddpm/checkpoints/epoch=000505.ckpt")[
            "state_dict"
        ],
        strict=True,
    )
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    sampler = DDIMSampler(model)

    # train
    train_average_ssim = 0
    train_average_mse = 0
    train_average_psnr = 0
    # test
    test_average_ssim = 0
    test_average_mse = 0
    test_average_psnr = 0
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    print(f"train dataset size: {len(train_loader.dataset)}")
    print(f"test dataset size: {len(test_loader.dataset)}")
    shape = (8, 60, 80)
    # with tqdm(train_loader, desc="calc train dataset ssim") as pbar:
    #     for i, data in enumerate(pbar):
    #         image = data["image"].to(device)
    #         segmentation = data["segmentation"].to(device)
    #         # reference = transform(np.zeros([480, 640, 3]), device=device)
    #         reference = data["reference"].to(device)
    #         # reference = transform(cv2.imread("data/vs_examples/00001-seg.jpg"), device)
    #         with torch.no_grad():
    #             with model.ema_scope():
    #                 c1 = model.cond_stage_model.encode(segmentation)
    #                 c2 = model.cond_stage_model.encode(reference)
    #                 # latent_image = model.first_stage_model.encode(image).mode()
    #                 c = torch.cat([c1, c2], dim=1)
    #                 sample, _ = sampler.sample(
    #                     S=opt.steps,
    #                     conditioning=c,
    #                     batch_size=c.shape[0],
    #                     shape=shape,
    #                     verbose=False,
    #                     eta=0,
    #                     x_T=torch.zeros((c.shape[0], *shape), device=device),
    #                 )
    #                 x_sample = model.first_stage_model.decode(sample)
    #         sample_show = inverse_transform(x_sample)
    #         image_show = inverse_transform(image)
    #         # cv2.imshow("sample", sample_show)
    #         # cv2.imshow("image", image_show)
    #         # diff_image = torch.abs(image - x_sample)
    #         # diff_image = torch.clip(diff_image * 0.5 + 0.5, 0, 1)
    #         # diff_image_show = inverse_transform(diff_image)
    #         # cv2.imshow("difference", diff_image_show)
    #         # cv2.waitKey(0)
    #         segmentation = torch.clip(segmentation * 0.5 + 0.5, 0, 1)
    #         image = torch.clip(image * 0.5 + 0.5, 0, 1)
    #         x_sample = torch.clip(x_sample * 0.5 + 0.5, 0, 1)
    #         train_average_ssim += piq.ssim(
    #             x_sample, image, data_range=1, reduction="sum"
    #         )
    #         train_average_mse += F.mse_loss(x_sample, image, reduction="sum")
    #         train_average_psnr += calc_psnr(x_sample, image, reduction="sum")
    #         pbar.set_postfix(
    #             ssim=(train_average_ssim / (i + 1) / batch_size).item(),
    #             mse=(train_average_mse / (i + 1) / batch_size).item(),
    #             psnr=(train_average_psnr / (i + 1) / batch_size),
    #         )
    # print(f"train average ssim: {train_average_ssim / len(train_loader.dataset)}")
    # print(f"train average mse: {train_average_mse / len(train_loader.dataset)}")
    # print(f"train average psnr: {train_average_psnr / len(train_loader.dataset)}")
    with tqdm(test_loader, desc="calc test dataset ssim") as pbar:
        for i, data in enumerate(pbar):
            image = data["image"].to(device)
            segmentation = data["segmentation"].to(device)
            reference = data["reference"].to(device)
            with torch.no_grad():
                with model.ema_scope():
                    c1 = model.cond_stage_model.encode(segmentation)
                    c2 = model.cond_stage_model.encode(reference)
                    c = torch.cat([c1, c2], dim=1)
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
            x_sample = torch.clip(x_sample * 0.5 + 0.5, 0, 1)
            image = torch.clip(image * 0.5 + 0.5, 0, 1)
            test_average_ssim += piq.ssim(
                x_sample, image, data_range=1, reduction="sum"
            )
            test_average_mse += torch.mean((x_sample * 255. - image * 255.) ** 2).item()
            test_average_psnr += calc_psnr(x_sample, image)
            pbar.set_postfix(
                ssim=(test_average_ssim / (i + 1) / batch_size).item(),
                mse=(test_average_mse / (i + 1) / batch_size),
                psnr=(test_average_psnr / (i + 1) / batch_size),
            )
            print(f"{(test_average_ssim / (i + 1) / batch_size).item()=}")
            print(f"{(test_average_mse / (i + 1) / batch_size)=}")
            print(f"{(test_average_psnr / (i + 1) / batch_size)=}")
    print(f"test average ssim: {test_average_ssim / len(test_loader.dataset)}")
    print(f"test average mse: {test_average_mse / len(test_loader.dataset)}")
    print(f"test average psnr: {test_average_psnr / len(test_loader.dataset)}")
