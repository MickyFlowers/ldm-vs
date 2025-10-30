import sys
import threading
import time
import queue

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
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion2
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dino_ddpm_v2 import DinoLatentDiffusion

# 全局变量用于线程间通信
reference_image = None
mask = None
generation_complete = False
stop_threads = False


def transform(img: np.ndarray, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch

def continuous_generation(model, sampler, device, input_reference, steps, result_queue, update_interval=10.0,camera=None, sam=None):
    """持续生成参考图像的线程函数"""
    start_time = time.time()
    global current_conditioning, stop_threads
    
    color_image, _ = camera.get_frame()
        
     # 使用SAM分割
    _, conditioning_img = sam.segment_img(color_image)
        
     # 转换为conditioning
    current_conditioning = transform(conditioning_img, device=device)
    c1 = model.cond_stage_model.encode(current_conditioning)
    c2 = model.cond_stage_model.encode(input_reference)
    c = torch.cat([c1, c2], dim=1)
    c = model.position_enc(c)
    
    shape = (8, 60, 80)
    sample, _ = sampler.sample(
        S=steps,
        conditioning=c,
        batch_size=c.shape[0],
        shape=shape,
        verbose=False,
        eta=0,
        x_T=torch.zeros((c.shape[0], *shape), device=device),
    )
    x_sample = model.first_stage_model.decode(sample)
    reference_image_tensor = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    reference_image_np = (
        reference_image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    )
    reference_image = cv2.cvtColor(reference_image_np, cv2.COLOR_RGB2BGR)
    reference_image = reference_image.astype(np.uint8)
    
    # 将生成的图像放入队列
    result_queue.put(('reference_image', reference_image))
    

    # 计算剩余等待时间
    elapsed = time.time() - start_time
    remaining = max(0.0, update_interval - elapsed)
    time.sleep(remaining)


def visual_servoing_loop(camera, ibvs_controller, roma_ibvs_controller, ur_robot, 
                         camera2tcp, use_roma, control_rate, result_queue, sam):
    """视觉伺服控制的主循环"""
    global reference_image, mask, generation_complete, stop_threads
    
    logging.info("Start Servoing...")
    
    # 控制循环
    while not stop_threads:
        
        # 检查是否有新的生成结果
        try:
            result_type, result_data = result_queue.get_nowait()
            if result_type == 'reference_image':
                # 更新参考图像和mask
                reference_image = result_data
                _, mask = sam.segment_img(reference_image)
                _, mask_background = sam.segment_img(reference_image)
                mask[mask_background == True] = True
                logging.info("Updated reference image")
                
            elif result_type == 'error':
                logging.error(f"Generation error: {result_data}")
                stop_threads = True
                return
                
        except queue.Empty:
            pass  # 没有新图像时继续使用当前参考图像
        
        # 获取当前帧
        color_image, depth_image = camera.get_frame()
        depth_image += 1e-5
        reference_depth = np.zeros_like(depth_image)
        reference_depth[:] = 0.5
        
        # 计算速度
        if use_roma:
            roma_ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = roma_ibvs_controller.calc_vel(mask=mask)
        else:
            ibvs_controller.update(
                cur_img=color_image,
                tar_img=reference_image,
                cur_depth=depth_image,
                tar_depth=reference_depth,
            )
            _, vel, score, match_img = ibvs_controller.calc_vel(mask=mask)
        
        logging.info(f"vel norm: {np.linalg.norm(vel)}, score: {score:.3f}")
        if score > 0.8:
            use_roma = True
        
        # 应用速度控制
        vel = linkVelTransform(vel, camera2tcp)  # 根据匹配分数调整速度
        if use_roma:
            ur_robot.applyTcpVel(vel * 0.2)
        else:
            ur_robot.applyTcpVel(vel * 0.2)
        
        # 显示匹配结果
        if match_img is not None:
            cv2.imshow("match_img", match_img)
        
        # 检查退出条件
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logging.info("Manual stop requested")
            break

def main():
    global stop_threads, reference_image, mask, generation_complete
    
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
        help="control frequency in Hz",
    )
    parser.add_argument(
        "--image_only",
        type=bool,
        default=False,
        help="only use image",
    )
    
    opt = parser.parse_args()
    camera = RealSenseCamera(exposure_time=700)
    
    if not opt.image_only:
        ip = "172.16.8.33"
        ur_robot = UR(ip)
        gripper = robotiq_gripper.RobotiqGripper()
        gripper.connect(ip, 63352)
        gripper_enable = False
        key = input("Activate gripper?: [Y/n]")
        if key.lower() == "y":
            gripper_enable = True
            gripper.activate()
    
    use_roma = True
    ibvs_controller = IBVS(camera, kp_algo=KpMatchAlgo, match_threshold=0.4)
    roma_ibvs_controller = IBVS(camera, kp_algo=RomaMatchAlgo)
    sam = SAM()
    
    config = OmegaConf.load("logs/2025-08-15T17-55-08_screwdriver-m/configs/2025-08-15T17-55-08-project.yaml")
    input_reference = cv2.imread("data/vs_examples/vs_finetune.jpg")
    
    model = instantiate_from_config(config.model)
    logging.info("Loading model...")
    
    model.load_state_dict(
        torch.load("logs/2025-08-15T17-55-08_screwdriver-m/checkpoints/epoch=000063.ckpt")[
            "state_dict"
        ],
        strict=True,
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    
    tcp_pose = np.load("data/vs_examples/extrinsic/vs_tcp_pose.npy")
    camera2tcp = np.load("calibration_data/left_arm/result/camera2tcp.npy")
    input_reference_tensor = transform(input_reference, device=device)
    
   
    
    # 用户交互部分
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
        # 创建线程通信队列
        result_queue = queue.Queue()
        
        # 启动持续生成线程
        generation_thread = threading.Thread(
            target=continuous_generation,
            args=(model, sampler, device, input_reference_tensor, opt.steps, result_queue, 2.0,camera,sam)
        )
        generation_thread.daemon = True
        generation_thread.start()
    
        time.sleep(5)
        # 启动视觉伺服控制
        try:
            visual_servoing_loop(
                camera, ibvs_controller, roma_ibvs_controller, ur_robot,
                camera2tcp, use_roma, opt.control_rate, result_queue, sam
            )
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"控制错误: {e}")
        finally:
            stop_threads = True
            if not opt.image_only:
                ur_robot.stop()
            cv2.destroyAllWindows()
            
               

if __name__ == "__main__":
    main()