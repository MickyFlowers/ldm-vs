import torchvision
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageFilter
import os
import random

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


class VsDatasetBase(data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.flist = self.get_data_flist()
        self.processer = self.create_processer()
        self.loader = pil_loader

    def get_data_flist(self):
        raise NotImplementedError

    def create_processer(self):
        raise NotImplementedError

    def __getitem__(self, index):
        ret = {}
        if self.seg:
            file_name = str(self.flist["segmentation"][index])
            image = self.loader(os.path.join(self.data_root, file_name))
            image = enhance_image(image)
            image = self.processer(image)
            ret["image"] = image
        else:
            img_file_name = str(self.flist["image"][index])
            seg_file_name = str(self.flist["segmentation"][index])
            image = self.loader(os.path.join(self.data_root, img_file_name) + ".jpg")
            segmentation = self.loader(
                os.path.join(self.data_root, seg_file_name) + ".jpg"
            )
            segmentation = enhance_image(segmentation)
            image = self.processer(image)
            segmentation = self.processer(segmentation)
            ret["image"] = image
            ret["segmentation"] = segmentation
        return ret

    def __len__(self):
        return len(self.flist["image"])


class VsDataset(VsDatasetBase):
    def __init__(
        self, data_flist=None, img_size=[480, 640], seg=False, **kwargs
    ) -> None:
        self.data_flist = data_flist
        self.img_size = img_size
        self.seg = seg
        super().__init__(**kwargs)

    def get_data_flist(self):
        data_flist = os.path.join(self.data_root, self.data_flist)
        flist = make_dataset(data_flist)
        flist_dict = {}

        img_flist = [os.path.join("img", file) for file in flist]
        seg_flist = [os.path.join("seg", file) for file in flist]
        flist_dict["image"] = img_flist
        flist_dict["segmentation"] = seg_flist

        return flist_dict

    def create_processer(self):
        tfs = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.img_size[0], self.img_size[1])),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        return tfs
