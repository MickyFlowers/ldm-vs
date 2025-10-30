import os
import cv2
from xlib.sam.sam_gui import SAM
import numpy as np

root_path = "/home/cyx/project/latent-diffusion/temp/video/backup/10101309/first_image"
# root_path = "./"

seg_pictures = [
    "000.jpg",
    # "000-023.jpg",
]
# seg_pictures = ["start.png"]
sam = SAM()
final_mask = None
for seg_picture in seg_pictures:
    file_path = os.path.join(root_path, seg_picture)
    image = cv2.imread(file_path)
    _, mask, _, _ = sam.segment_img(image)
    # for i in range(3):
    #     _, mask, _, _ = sam.segment_img(image)
    #     if final_mask is None:
    #         final_mask = mask
    #     else:
    #         final_mask = final_mask | mask
    seg = np.ones_like(image) * 255
    seg[mask] = image[mask]
    cv2.imwrite(os.path.join(root_path, "seg" + seg_picture), seg)
