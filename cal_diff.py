# import cv2
# import lpips
# import torch
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# import numpy as np
# from torchvision.transforms.functional import to_tensor, resize
# from PIL import Image

# input = cv2.imread("logs/2024-12-25T14-08-32_dino_ddpm_single/images/val/inputs_gs-051249_e-000204_b-000065.png")
# rec = cv2.imread("logs/2024-12-25T14-08-32_dino_ddpm_single/images/val/reconstruction_gs-051249_e-000204_b-000065.png")
# sample = cv2.imread("logs/2024-12-25T14-08-32_dino_ddpm_single/images/val/samples_gs-051249_e-000204_b-000065.png")

# # # diff_input_rec = cv2.absdiff(input, rec)
# diff_input_sample = cv2.absdiff(input, sample)
# # # diff_rec_sample = cv2.absdiff(rec, sample)
# cv2.imshow("input",input)
# cv2.imshow("sample",sample)
# # # cv2.imshow("diff_input_rec",diff_input_rec)
# cv2.imshow("diff_input_sample",diff_input_sample)
# # # cv2.imshow("diff_rec_sample",diff_rec_sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 转换为 RGB 格式
# input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
# rec= cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
# from skimage.metrics import structural_similarity as ssim
# # 计算 SSIM
# ssim_score, diff = ssim(input,rec, full=True, channel_axis=-1)



# def compute_psnr(img1, img2):
#     """
#     img1, img2: numpy arrays of shape (H, W, C), dtype float32, range [0, 1]
#     """
#     return compare_psnr(img1, img2, data_range=1.0)


# # Load LPIPS model (AlexNet backbone; also supports VGG and Squeeze)
# lpips_fn = lpips.LPIPS(net='alex').cuda()

# def compute_lpips(img1_tensor, img2_tensor):
#     """
#     img1_tensor, img2_tensor: torch tensors of shape (1, 3, H, W), range [-1, 1]
#     """
#     with torch.no_grad():
#         d = lpips_fn(img1_tensor, img2_tensor)
#     return d.item()



# # 读取和预处理图像（假设你有 image1_path, image2_path）
# def load_and_preprocess(path):
#     img = Image.open(path).convert("RGB")
#     img = resize(img, (256, 256))
#     img_tensor = to_tensor(img).unsqueeze(0)  # (1, 3, H, W), [0, 1]
#     img_tensor_lpips = img_tensor * 2 - 1     # 转换到 [-1, 1] 给 LPIPS 用
#     return img_tensor, img_tensor_lpips, np.array(img).astype(np.float32) / 255.0

# # 示例
# img1_tensor, img1_lpips, img1_np = load_and_preprocess("logs/2024-12-25T14-08-32_dino_ddpm_single/images/val/inputs_gs-051249_e-000204_b-000065.png")
# img2_tensor, img2_lpips, img2_np = load_and_preprocess("logs/2024-12-25T14-08-32_dino_ddpm_single/images/val/samples_gs-051249_e-000204_b-000065.png")

# # 计算指标
# psnr = compute_psnr(img1_np, img2_np)
# lpips_score = compute_lpips(img1_lpips.cuda(), img2_lpips.cuda())

# #输出指标
# print(f"SSIM score: {ssim_score:.4f}")
# print(f"psnr score: {psnr:.4f}")
# print(f"lpips_score score: {lpips_score:.4f}")



# # SSIM score: 0.9433
# # psnr score: 42.5090
# # lpips_score score: 0.0013

# # SSIM score: 0.9407
# # psnr score: 43.5828
# # lpips_score score: 0.0010

# # 53epoch
# # SSIM score: 0.9463
# # psnr score: 44.5676
# # lpips_score score: 0.0007
import sys

sys.path.append("./")
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

config = OmegaConf.load(
    "logs/2025-07-10T01-46-20_dino_ddpm_charge/configs/2025-07-10T01-46-20-project.yaml"
)
model: DinoLatentDiffusionSingle = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load(
        "logs/2025-07-10T01-46-20_dino_ddpm_charge/checkpoints/fintune_epoch=000024.ckpt"
    )["state_dict"],
    strict=True,
) 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = DDIMSampler(model)
def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch
conditioning=cv2.imread("/mnt/workspace/cyxovo/dataset/charge_data_single/seg/00008.jpg")
conditioning = transform(conditioning, device=device)
c = model.cond_stage_model.encode(conditioning)
shape = (8, 60, 80)
reference_image, _ = sampler.sample(
    S=50,
    conditioning=c,
    batch_size=c.shape[0],
    shape=shape,
    verbose=False,
    eta=0,
    x_T=torch.zeros((c.shape[0], *shape), device=device),
)
x_sample = model.first_stage_model.decode(reference_image)
reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
reference_image = (
    reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
)
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
reference_image = reference_image.astype(np.uint8)

input = cv2.imread("/mnt/workspace/cyxovo/dataset/charge_data_single/img/00008.jpg")
 
diff_input_sample = cv2.absdiff(input, reference_image)

cv2.imshow("input",input)
cv2.imshow("sample",reference_image)
cv2.imshow("diff_input_sample",diff_input_sample)

cv2.waitKey(0)
cv2.destroyAllWindows()

# # 转换为 RGB 格式
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
rec= cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
from skimage.metrics import structural_similarity as ssim
# 计算 SSIM
ssim_score, diff = ssim(input,rec, full=True, channel_axis=-1)
print(ssim_score)