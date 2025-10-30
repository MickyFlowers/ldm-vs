import os
import random

root_path = "/mnt/workspace/cyxovo/dataset/20250912_charge_m"
image_path = os.path.join(root_path, "img")
image_files = os.listdir(image_path)
image_files.sort()
num_images = len(image_files)
num_val = int(num_images * 0.1)
val_index = random.sample(range(num_images), num_val)
train_index = list(set(range(num_images)) - set(val_index))
val_index.sort()
train_index.sort()
train_images = [image_files[i] for i in train_index]
val_images = [image_files[i] for i in val_index]
with open(os.path.join(root_path, "train_img.flist"), "w") as f:
    for train_image in train_images:
        f.write("img/" + train_image + "\n")

with open(os.path.join(root_path, "val_img.flist"), "w") as f:
    for val_image in val_images:
        f.write("img/" + val_image + "\n")