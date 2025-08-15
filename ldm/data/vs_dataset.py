import torchvision
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageFilter
import os
import random
import torch

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
    
class VaeDataset(data.Dataset):
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
        filename = self.flist[index]
        gt_img_file_name = os.path.join(self.data_root, filename)
        gt_image = self.loader(gt_img_file_name)
        gt_image = self.processer(gt_image)
        ret["image"] = gt_image
        return ret

    def __len__(self):
        return len(self.flist)
    
class LdmDataset(data.Dataset):
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
        img_wrapper = self.flist[index]
        img_wrapper_list= img_wrapper.split(",")

        gt_count_file_name = os.path.join(self.data_root, img_wrapper_list[0])
        ref_img_file_name = os.path.join(self.data_root, img_wrapper_list[1])
        peg_count_file_name = os.path.join(self.data_root, img_wrapper_list[2])
        
        ref_img = self.loader(ref_img_file_name)
        peg_img = self.loader(peg_count_file_name)
        gt_image = self.loader(gt_count_file_name)
        ref_img = enhance_image(ref_img)
        peg_img = enhance_image(peg_img)
        ref_img = self.processer(ref_img)
        peg_img = self.processer(peg_img)
        gt_image = self.processer(gt_image)
        ret["image"] = gt_image
        ret["segmentation"] = peg_img
        ret["reference"] = ref_img
        return ret

    def __len__(self):
        return len(self.flist)
