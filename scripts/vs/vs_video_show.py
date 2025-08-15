import sys

sys.path.append("./")
#print(sys.path)
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
parser.add_argument(
    "--image_only",
    type=bool,
    default=False,
    help="only use image",
)



opt = parser.parse_args()
camera = RealSenseCamera(exposure_time=600)
# camera.set_param(camera.fx/640.0*256.0, camera.fy/480.0*256.0,camera.cx/640.0*256.0,camera.cy/480.0*256.0)
if not opt.image_only:
    ip = "172.16.8.33"
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
# root_path = os.path.dirname(os.path.realpath(__file__))


# config = OmegaConf.load(
#     "logs/dino_Dec_19_20_26/configs/2024-12-18T09-31-31-project.yaml"
# )


config = OmegaConf.load("logs/2025-06-29-dino-ddpm-finetune/2025-07-01T01-52-41-project.yaml")
id = "031-001"
# input_reference = cv2.imread(
#    f"/home/cyx/project/data/good_random_data_single/img/{id}.jpg"
# )
input_reference = cv2.imread("data/vs_examples/vs_finetune.jpg")
# input_reference = cv2.resize(input_reference, (256, 256), interpolation=cv2.INTER_AREA)
# cv2.imshow("input_reference", input_reference)
# cv2.waitKey(0)
# model: LatentDiffusion2 = instantiate_from_config(config.model)

model:DinoLatentDiffusion=instantiate_from_config(config.model)
logging.info("Loading model...")


model.load_state_dict(
    torch.load("logs/2025-06-29-dino-ddpm-finetune/epoch=000031.ckpt")[
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

# tcp_pose=np.array([[-0.62897465, -0.19662884 ,-0.75214891 , 0.21489352],
#           [-0.77090209  ,0.03267384 , 0.63611507,  0.59392125],
#           [-0.10050298  ,0.97993343, -0.17213261, -0.23139775],
#            [ 0.         , 0.      ,    0.    , 1.        ]])
camera2tcp = np.load("calibration_data/left_arm/result/camera2tcp.npy")
input_reference = transform(input_reference, device=device)


# 设置视频写入器
target_width = 1280  # 与match_images同宽
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
fps = 10
video_size = (target_width, 480 * 2)  # 总高度=上方480 + 下方480
out = cv2.VideoWriter('sam_aligned_output.mp4', fourcc, fps, video_size)

# 初始化SAM处理
def visualize_mask(image, mask):
    """可视化mask叠加效果"""
    bw_image = np.zeros(image.shape[:2], dtype=np.uint8)
    bw_image[mask] = 255
    return cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)


while True:
    key =input("End servo process?: [Y/n]")
    if key.lower() == "y":
        break
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
        if not opt.image_only:
            while count < 30:
                color_image, depth_image = camera.get_frame()
                count += 1
            color_image, depth_image = camera.get_frame()
        
        
        conditioning, init_mask = sam.segment_img(color_image)
        
        conditioning = transform(conditioning, device=device)
        
        
        
        with torch.no_grad():
            with model.ema_scope():
                c1 = model.cond_stage_model.encode(conditioning)
                # c2 = model.first_stage_model.encode(input_reference)
                c2 = model.cond_stage_model.encode(input_reference)
                
                
                # print(type(c1), type(c2))
                # c1=c1.values
                print(c1.shape, c2.shape) 
                c = torch.cat([c1, c2], dim=1)
                c= model.position_enc(c)
                # print(c.shape)
                
               
                shape = (8, 60, 80)
                # shape = (3, 64, 64)
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
                reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                reference_image = (
                    reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                )
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
                reference_image = reference_image.astype(np.uint8)
                
              
                _, mask1 = sam.segment_img(reference_image)
        
                # 第二次SAM处理（细化）
                _, mask_background = sam.segment_img(reference_image)
                final_mask = mask1.copy()
                final_mask[mask_background == True] = True 
                
                  # 1. 上方三窗格 (调整为1280宽度)
                h, w = reference_image.shape[:2]
                scale = target_width / (w*3)  # 计算缩放因子使三窗格总宽=1280
                new_h, new_w = int(h*scale), int(w*scale)
                
                top_frame = np.zeros((480, target_width, 3), dtype=np.uint8)
                
                # 窗格1: 初始mask (彩色)
                resized_init = cv2.resize(visualize_mask(reference_image, init_mask), 
                                        (new_w, new_h))
                top_frame[:new_h, :new_w] = resized_init
                
                # 窗格2: SAM_background结果 (黑白)
                resized_mask1 = cv2.resize(visualize_mask(reference_image, mask_background), 
                                        (new_w, new_h))
                top_frame[:new_h, new_w:2*new_w] = resized_mask1
                
                # 窗格3: 最终结果 (彩色)
                resized_final = cv2.resize(visualize_mask(reference_image, final_mask), 
                                        (new_w, new_h))
                top_frame[:new_h, 2*new_w:3*new_w] = resized_final
                
                # 添加标签 (白色文字，黑色描边更清晰)
                font = cv2.FONT_HERSHEY_SIMPLEX
                def put_text_with_outline(img, text, pos):
                    cv2.putText(img, text, pos, font, 0.6, (0,0,0), 4, cv2.LINE_AA)
                    cv2.putText(img, text, pos, font, 0.6, (255,255,255), 2, cv2.LINE_AA)
                
                put_text_with_outline(top_frame, "front", (20, 30))
                put_text_with_outline(top_frame, "background", (new_w+20, 30))
                put_text_with_outline(top_frame, "Final", (2*new_w+20, 30))
                        
    # else:
    #     logging.error("please segment the image to continue")
    #     exit()
    logging.info("Start Servoing...")
    use_roma = True
    
    while True:
        color_image, depth_image = camera.get_frame()
        depth_image += 1e-5
        reference_depth = np.zeros_like(depth_image)
        reference_depth[:] = 0.5
        if use_roma:
            roma_ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = roma_ibvs_controller.calc_vel(mask=final_mask)
        else:
            ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = ibvs_controller.calc_vel(mask=final_mask)
        logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
        if score > 0.8:
            use_roma = True
        vel = linkVelTransform(vel, camera2tcp)
        if use_roma:
            ur_robot.applyTcpVel(vel * 0.2)
        else:
            ur_robot.applyTcpVel(vel * 0.2)
        cv2.imshow("match_img", match_img)
         # 2. 下方match_images (保持1280x480原尺寸)
        bottom_frame = np.zeros((480, target_width, 3), dtype=np.uint8)
        bottom_frame[:480, :1280] = match_img  # 直接放入原图
        
        # 添加标题 
        cv2.putText(bottom_frame, "Match Images", (20, 40), 
                   font, 0.8, (0,0,0), 6, cv2.LINE_AA)
        cv2.putText(bottom_frame, "Match Images", (20, 40), 
                   font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # 合并上下两部分
        combined = np.vstack([top_frame, bottom_frame])
        
        # 写入视频帧
        out.write(combined)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ur_robot.stop()
            cv2.destroyAllWindows()
            break
        # time.sleep(1 / opt.control_rate)
out.release()
    
    # while not opt.image_only:
    #     color_image, depth_image = camera.get_frame()
    #     # conditioning = np.zeros_like(color_image)
    #     # conditioning[mask] = color_image[mask]
    #     # conditioning = transform(conditioning, device=device)
    #     # with torch.no_grad():
    #     #     with model.ema_scope():
    #     #         c = model.cond_stage_model.encode(conditioning).mode()
    #     #         sample, _ = sampler.sample(
    #     #             S=opt.steps,
    #     #             conditioning=c,
    #     #             batch_size=c.shape[0],
    #     #             shape=c.shape[1:],
    #     #             verbose=False,
    #     #             x_T=torch.zeros(c.shape, device=device),
    #     #         )
    #     #         x_sample = model.first_stage_model.decode(sample)
    #     #         reference_image = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    #     #         reference_image = (
    #     #             reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    #     #         )
    #     #         reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
    #     #         reference_image = reference_image.astype(np.uint8)
    #     depth_image += 1e-5
    #     reference_depth = np.zeros_like(depth_image)
    #     reference_depth[:] = 0.5
    #     if use_roma:
    #         vel, score, match_img = roma_ibvs_controller.cal_vel_from_img(
    #             reference_image,
    #             color_image,
    #             reference_depth,
    #             depth_image,
    #             mask,
    #             True,
    #         )
    #     else:
    #         vel, score, match_img = ibvs_controller.cal_vel_from_img(
    #             reference_image,
    #             color_image,
    #             reference_depth,
    #             depth_image,
    #             mask,
    #             True,
    #         )
    #     logging.info(f"vel norm: {np.linalg.norm(vel)}, {vel=}")
    #     if np.linalg.norm(vel) < 0.15:
    #         use_roma = False
    #     vel = linkVelTransform(vel, camera2tcp)
    #     if use_roma:
    #         ur_robot.applyTcpVel(vel * 0.15)
    #     else:
    #         ur_robot.applyTcpVel(vel * 0.5)
    #     cv2.imshow("match_img", match_img)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         ur_robot.stop()
    #         cv2.destroyAllWindows()
    #         break
    #     # time.sleep(1 / opt.control_rate)
