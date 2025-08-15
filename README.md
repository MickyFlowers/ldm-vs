# 扩散模型任务说明

## 模型简介

详情查看论文：https://arxiv.org/abs/2112.10752

该代码完全使用pytorch_lightning进行训练，可以直接跳过训练部分，直接看模型文件

ldm/models/diffusion/dino_ddpm.py latent-diffusion

ldm/models/autoencoder.py 变分自编码器

## 文件结构

```
├── assets # 源代码的example
├── calibration_data #用于存放相机的标定数据与标定结果
├── configs# 模型训练的配置文件
├── data # 一些example data
├── environment.yaml # anaconda环境
├── ldm # 存放模型的主要文件夹
├── log # 存放训练日志
├── main.py # 训练启动代码
├── models # 官方源码给出的checkpoint
├── scripts #脚本
```

## 快速开始

```bash
CUDA_VISIABLE_DEVICES=0,1 nohup python main.py --base configs/autoencoder/screwdriver_sim.yaml -t --gpus 0,1 --max_epochs 1000 > log.log 2>&1 &
```

参数解释：CUDA_VISIABLE_DEVICES和--gpus用于指定GPU的ID，用于多卡训练，--base参数用于指定训练用的配置文件。nohup则是将训练放在后台并且将日志文件输出到当前目录的log.log文件夹当中。

### 配置文件说明

configs/autoencoder文件夹下存放的是用于将图像编码成latent的变分自编码器。

configs/ldm文件夹下存放的是训练latent-diffusion的配置文件。

在服务器上，vs_random_img.yaml是训练自编码器的配置文件，Dino_ddpm是将condition进行DINOv2编码并训练latent-diffusion的配置文件，将文件替换即可。

## 其他脚本

scripts/vs/get_random_data.py 实物采集数据

scripts/vs/generate_flist.py 生成训练数据的配置文件

scripts/vs/rename_random_data.py 数据处理，用于给sam2做分割

scripts/vs/vs.py 实物实验部署代码

scripts/calc_average_ssim.py 计算mse，psnr和ssim值

scripts/calibrate.py 标定相机外参

scripts/test文件夹中主要用于查看数据集中图像的生成效果





