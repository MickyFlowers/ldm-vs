import sys

sys.path.append("./")
from ldm.wrapper import InferenceWrapper
import torch
import cv2
import numpy as np
from xlib.algo.vs.vs_controller.ibvs import IBVS
from xlib.algo.vs.kp_matcher import RomaMatchAlgo, KpMatchAlgo
from xlib.device.sensor.camera import RealSenseCamera
from xlib.sam.sam_gui import SAM

camera = RealSenseCamera(exposure_time=500)
# roma_ibvs_controller = IBVS(camera, kp_algo=KpMatchAlgo, match_threshold=0.9)
roma_ibvs_controller = IBVS(camera, kp_algo=RomaMatchAlgo)
config_path = "models/ldm/ddpm_vs/2025-01-14T12-18-48-project.yaml"
ckpt_path = "models/ldm/ddpm_vs/last.ckpt"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = InferenceWrapper(config_path, ckpt_path, device=device)
sam = SAM()


conditioning = cv2.imread(
    f"/mnt/workspace/cyxovo/dataset/good_random_data_single/seg/030-003.jpg"
)
for i in range(10):
    input_reference = cv2.imread(
        f"/mnt/workspace/cyxovo/dataset/good_random_data_single/img/000-{i:03d}.jpg"
    )
    input_reference_compare = cv2.imread(
        f"/mnt/workspace/cyxovo/dataset/good_random_data_single/img/000-{(i+1):03d}.jpg"
    )
    output = model.inference(input_reference, conditioning)
    output_compare = model.inference(input_reference_compare, conditioning)
    img_show = np.concatenate(
        [output, input_reference, np.abs(output - output_compare)], axis=1
    )
    _, mask = sam.segment_img(input_reference)
    reference_depth = np.full(output.shape[:2], 0.3)
    depth_image = np.full(output.shape[:2], 0.5)
    roma_ibvs_controller.update(
        cur_img=input_reference,
        tar_img=output,
        cur_depth=depth_image,
        tar_depth=reference_depth,
    )
    _, vel, score, match_img = roma_ibvs_controller.calc_vel(mask=mask)
    
    # print(match_img.shape)
    cv2.imshow("match_img", match_img)
    cv2.imshow("ref|peg|out", img_show)
    cv2.waitKey(0)
