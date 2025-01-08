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
from ldm.models.diffusion.ddpm import LatentDiffusion2
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

camera = RealSenseCamera(exposure_time=500)
ip = "172.16.11.33"
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
use_roma = True
ibvs_controller = IBVS(camera, kp_algo=KpMatchAlgo, match_threshold=0.4)
roma_ibvs_controller = IBVS(camera, kp_algo=RomaMatchAlgo)
sam = SAM()
root_path = os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(
    "logs/dino_Dec_19_20_26/configs/2024-12-18T09-31-31-project.yaml"
)
id = "031-001"
# input_reference = cv2.imread(
#    f"/home/cyx/project/data/good_random_data_single/img/{id}.jpg"
# )
input_reference = cv2.imread("data/vs_examples/reference.png")
cv2.imshow("input_reference", input_reference)
cv2.waitKey(0)
model: LatentDiffusion2 = instantiate_from_config(config.model)
logging.info("Loading model...")
model.load_state_dict(
    torch.load("logs/dino_Dec_19_20_26/checkpoints/epoch=000011.ckpt")[
        "state_dict"
    ],
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
        count = 0
        while count < 30:
            color_image, depth_image = camera.get_frame()
            count += 1
        color_image, depth_image = camera.get_frame()
        # conditioning = cv2.imread(
        #     f"/home/cyx/project/data/good_random_data_single/seg/{id}.jpg"
        # )
        conditioning, mask = sam.segment_img(color_image)
        # conditioning = cv2.imread("/home/cyx/Pictures/05942.jpg")
        conditioning = transform(conditioning, device=device)
        with torch.no_grad():
            with model.ema_scope():
                c1 = model.cond_stage_model.encode(conditioning).mode()
                c2 = model.first_stage_model.encode(input_reference).mode()
                c = torch.cat([c1, c2], dim=1)
                sample, _ = sampler.sample(
                    S=opt.steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=c.shape[1:],
                    verbose=False,
                    eta=0,
                    x_T=torch.zeros(c1.shape, device=device),
                )
                x_sample = model.first_stage_model.decode(sample)
                reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                reference_image = (
                    reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                )
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
                reference_image = reference_image.astype(np.uint8)
                cv2.imshow("reference_image", reference_image)
                cv2.waitKey(0)
    else:
        logging.error("please segment the image to continue")
        exit()
    logging.info("Start Servoing...")
    use_roma = True
    while True:
        color_image, depth_image = camera.get_frame()
        # conditioning = np.zeros_like(color_image)
        # conditioning[mask] = color_image[mask]
        # conditioning = transform(conditioning, device=device)
        # with torch.no_grad():
        #     with model.ema_scope():
        #         c = model.cond_stage_model.encode(conditioning).mode()
        #         sample, _ = sampler.sample(
        #             S=opt.steps,
        #             conditioning=c,
        #             batch_size=c.shape[0],
        #             shape=c.shape[1:],
        #             verbose=False,
        #             x_T=torch.zeros(c.shape, device=device),
        #         )
        #         x_sample = model.first_stage_model.decode(sample)
        #         reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        #         reference_image = (
        #             reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        #         )
        #         reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
        #         reference_image = reference_image.astype(np.uint8)
        depth_image += 1e-5
        reference_depth = np.zeros_like(depth_image)
        reference_depth[:] = 0.5
        if use_roma:
            vel, score, match_img = roma_ibvs_controller.cal_vel_from_img(
                reference_image,
                color_image,
                reference_depth,
                depth_image,
                mask,
                True,
            )
        else:
            vel, score, match_img = ibvs_controller.cal_vel_from_img(
                reference_image,
                color_image,
                reference_depth,
                depth_image,
                mask,
                True,
            )
        logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
        if np.linalg.norm(vel) < 0.15:
            use_roma = False
        vel = linkVelTransform(vel, camera2tcp)
        if use_roma:
            ur_robot.applyTcpVel(vel * 0.15)
        else:
            ur_robot.applyTcpVel(vel * 0.5)
        cv2.imshow("match_img", match_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ur_robot.stop()
            cv2.destroyAllWindows()
            break
        # time.sleep(1 / opt.control_rate)
