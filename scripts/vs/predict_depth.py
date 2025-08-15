import torch
import numpy as np
from PIL import Image
import cv2
# Zoe_N
model_zoe_n= torch.hub.load("./ZoeDepth", "ZoeD_N", source="local", pretrained=True,trust_repo=True, skip_validation=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

image = Image.open("/mnt/workspace/cyxovo/dataset/charge_data_depth_single/img/00001.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

gt_image_array = cv2.imread("/mnt/workspace/cyxovo/dataset/charge_data_depth_single/depth/00001.png", cv2.IMREAD_ANYDEPTH)

depth_pil = zoe.infer_pil(image, output_type="pil")
print(f"Depth range: {depth_numpy.min()} to {depth_numpy.max()}")
print(gt_image_array.dtype)
print(f"gt_Depth range: {gt_image_array.min()} to {gt_image_array.max()}")
depth_pil.save("predict_depth.png")
