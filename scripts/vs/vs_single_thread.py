import sys

sys.path.append("./")
# print(sys.path)
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
from ldm.models.diffusion.dino_ddpm_v2 import DinoLatentDiffusion
import threading
import time
import copy


def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


class VsMutliThread:
    def __init__(self, args):
        self.args = args
        self.enable_vs = False
        self.camera = RealSenseCamera(
            exposure_time=args.exposure_time, serial_number="f1421776"
        )
        # self.camera.set_param(
        #     self.camera.fx * args.scale,
        #     self.camera.fy * args.scale,
        #     self.camera.cx * args.scale,
        #     self.camera.cy * args.scale,
        # )
        if not opt.image_only:
            left_base_to_world = np.load(
                "data/vs_examples/extrinsic/left_base_to_world.npy"
            )
            self.ur_robot = UR(args.ip, left_base_to_world)
            # activate gripper
            gripper = robotiq_gripper.RobotiqGripper()
            gripper.connect(self.args.ip, 63352)
            gripper_enable = False
            key = input("Activate gripper?: [Y/n]")
            if key.lower() == "y":
                gripper_enable = True

            if gripper_enable:
                gripper.activate()

        self.ibvs_controller = IBVS(
            self.camera, kp_algo=KpMatchAlgo, match_threshold=0.4
        )
        self.roma_ibvs_controller = IBVS(self.camera, kp_algo=RomaMatchAlgo)
        self.sam = SAM()
        config = OmegaConf.load(args.config)
        self.model: DinoLatentDiffusion = instantiate_from_config(config.model)
        self.model.load_state_dict(
            torch.load(args.ckpt_path)["state_dict"],
            strict=True,
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)
        self.tcp_pose = np.load(args.tcp_pose_file)
        self.camera2tcp = np.load(args.camera2tcp_file)
        self.control_thread = threading.Thread(target=self.control_thread_func)
        self.seg_thread = threading.Thread(target=self.seg_thread_func)
        self.lock = threading.Lock()
        self.first_mask = False
        key = input("Move to initial pose?: [Y/n]")
        if key.lower() == "y":
            self.ur_robot.moveToPose(self.tcp_pose, vel=0.1, acc=0.1)

        if gripper_enable:
            key = input("Open gripper?: [Y/n]")
            if key.lower() == "y":
                gripper.move_and_wait_for_pos(0, 255, 100)
            key = input("Close gripper?: [Y/n]")
            if key.lower() == "y":
                gripper.move_and_wait_for_pos(255, 255, 100)
        key = input("Capture image?: [Y/n]")
        if key.lower() == "n":
            # 获取当前TCP位姿（世界坐标系下）
            cur_pose = self.ur_robot.world_pose
            current_pos = cur_pose[:3, 3]  # TCP当前位置 [x, y, z]
            current_rot = cur_pose[:3, :3]  # TCP当前姿态（旋转矩阵）

            # 获取TCP的Z轴方向（世界坐标系下的当前朝向）
            tcp_z_axis = current_rot[:, 2]  # 旋转矩阵的第3列

            # 计算当前朝向的水平分量（在XY平面的投影）
            horizontal_dir = np.array([tcp_z_axis[0], tcp_z_axis[1], 0])
            if np.linalg.norm(horizontal_dir) > 0:
                horizontal_dir = horizontal_dir / np.linalg.norm(horizontal_dir)

            # 定义俯仰角（45度斜向下）越小越缓
            pitch_deg = 43
            pitch_rad = np.deg2rad(pitch_deg)

            # 计算目标方向向量（保持水平方向）构造一个单位向量，并且与世界坐标系z轴夹角为--45度
            target_dir = np.array(
                [
                    horizontal_dir[0] * np.cos(pitch_rad),
                    horizontal_dir[1] * np.cos(pitch_rad),
                    -np.sin(pitch_rad),  # 负号表示向下
                ]
            )

            # 移动距离
            move_distance = 0.10  # 5cm
            target_pos = current_pos + target_dir * move_distance
            # 构建目标位姿（保持当前姿态）
            target_pose = np.eye(4)
            target_pose[:3, :3] = current_rot  # 旋转部分不变
            target_pose[:3, 3] = target_pos  # 更新位置
            self.ur_robot.moveToWorldPose(target_pose, 0.1, 1)
        elif key.lower() == "y":
            self.init()
            self.control_thread.start()
            self.seg_thread.start()

    def init(self):
        if not opt.image_only:
            count = 0
            while count < 30:
                self.camera.get_frame()
                count += 1
            color_image, depth_image = self.camera.get_frame()
            cv2.imwrite("input_seg.jpg", color_image)
            conditioning, mask, self.points, self.background_points = (
                self.sam.segment_img(color_image)
            )
            conditioning = transform(conditioning, device=self.device)
            with torch.no_grad():
                with self.model.ema_scope():
                    c = self.model.cond_stage_model.encode(conditioning)
                    shape = (8, 60, 80)
                    sample, _ = self.sampler.sample(
                        S=self.args.steps,
                        conditioning=c,
                        batch_size=c.shape[0],
                        shape=shape,
                        verbose=False,
                        eta=0,
                        x_T=torch.zeros((c.shape[0], *shape), device=self.device),
                    )
                    x_sample = self.model.first_stage_model.decode(sample)
                    reference_image = torch.clamp(
                        (x_sample + 1.0) / 2.0, min=0.0, max=1.0
                    )
                    reference_image = (
                        reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    )
                    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
                    self.reference_image = reference_image.astype(np.uint8)
                    _, self.mask, self.peg_points, self.peg_background_points = (
                        self.sam.segment_img(self.reference_image)
                    )
                    (
                        _,
                        background_mask,
                        self.background_points,
                        self.background_background_points,
                    ) = self.sam.segment_img(self.reference_image)
                    self.mask[background_mask == True] = True

    def control_thread_func(self):
        start_time = time.time()
        num_threshold = 0
        while True:
            if self.enable_vs:
                color_image, depth_image = self.camera.get_frame()
                with self.lock:
                    self.color_image = copy.deepcopy(color_image)
                    self.first_mask = True

                depth_image += 1e-5
                reference_depth = np.zeros_like(depth_image)
                reference_depth[:] = 0.2
                with self.lock:
                    if self.args.use_roma:

                        self.roma_ibvs_controller.update(
                            cur_img=color_image,
                            tar_img=self.reference_image,
                            cur_depth=depth_image,
                            tar_depth=reference_depth,
                        )
                        _, vel, score, match_img = self.roma_ibvs_controller.calc_vel(
                            mask=self.mask
                        )
                    else:
                        self.ibvs_controller.update(
                            cur_img=color_image,
                            tar_img=self.reference_image,
                            cur_depth=depth_image,
                            tar_depth=reference_depth,
                        )
                        _, vel, score, match_img = self.ibvs_controller.calc_vel(
                            mask=self.mask
                        )
                if vel is None:
                    continue
                logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
                # if score > 0.78:
                #     self.args.use_roma = False
                if np.linalg.norm(vel * 0.3) < self.args.threshold:
                    num_threshold += 1
                    if num_threshold >= 15:
                        now_time = time.time()
                        print("total_time:", now_time - start_time)
                        self.ur_robot.stop()
                        break
                else:
                    num_threshold = 0
                vel = linkVelTransform(vel, self.camera2tcp)
                if self.args.use_roma:
                    self.ur_robot.applyTcpVel(vel * 0.3)
                else:
                    self.ur_robot.applyTcpVel(vel * 0.15)
                cv2.imshow("match_img", match_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    now_time = time.time()
                    print("total_time:", now_time - start_time)
                    self.ur_robot.stop()
                    cv2.destroyAllWindows()
                    break

    def seg_thread_func(self):
        while True:
            time.sleep(20.0)
            if self.enable_vs and self.first_mask:
                with self.lock:
                    color_image = self.color_image
                conditioning, mask = self.sam.segment_img_from_points(
                    color_image, self.points, self.background_points
                )
                conditioning = transform(conditioning, device=self.device)
                with torch.no_grad():
                    with self.model.ema_scope():
                        c = self.model.cond_stage_model.encode(conditioning)
                        shape = (8, 60, 80)
                        sample, _ = self.sampler.sample(
                            S=self.args.steps,
                            conditioning=c,
                            batch_size=c.shape[0],
                            shape=shape,
                            verbose=False,
                            eta=0,
                            x_T=torch.zeros((c.shape[0], *shape), device=self.device),
                        )
                        x_sample = self.model.first_stage_model.decode(sample)
                        reference_image = torch.clamp(
                            (x_sample + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        reference_image = (
                            reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                        )
                        reference_image = cv2.cvtColor(
                            reference_image, cv2.COLOR_RGB2BGR
                        )
                        with self.lock:
                            self.reference_image = reference_image.astype(np.uint8)
                            _, self.mask = self.sam.segment_img_from_points(
                                self.reference_image,
                                self.peg_points,
                                self.peg_background_points,
                            )
                            _, background_mask = self.sam.segment_img_from_points(
                                self.reference_image,
                                self.background_points,
                                self.background_background_points,
                            )
                            self.mask[background_mask == True] = True

    def start(self):
        self.enable_vs = True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--exposure_time",
    type=int,
    default=700,
    help="exposure time",
)
parser.add_argument(
    "--scale",
    type=float,
    default=1,
    help="image size scale",
)
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
parser.add_argument(
    "--image_only",
    type=bool,
    default=False,
    help="only use image",
)
parser.add_argument(
    "--ip",
    type=str,
    default="172.16.11.33",
    help="ip address of robot",
)
parser.add_argument(
    "--use_roma",
    type=bool,
    default=True,
    help="use roma",
)
parser.add_argument(
    "--config",
    type=str,
    default="logs/2025-08-26T01-38-46_dino_ddpm_charge/configs/2025-08-26T01-38-46-project.yaml",
    help="config file",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="logs/2025-08-26T01-38-46_dino_ddpm_charge/checkpoints/epoch=000373.ckpt",
    help="ckpt path",
)
parser.add_argument(
    "--tcp_pose_file",
    type=str,
    default="data/vs_examples/extrinsic/vs_tcp_pose_charge.npy",
    help="tcp pose file",
)
parser.add_argument(
    "--camera2tcp_file",
    type=str,
    default="calibration_data/left_arm/result/camera2tcp.npy",
    help="camera2tcp file",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.002,
    help="stop threshold",
)
opt = parser.parse_args()

vs = VsMutliThread(opt)
vs.start()
