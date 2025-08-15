from xlib.sam.sam_gui import SAM

import cv2  
sam = SAM()
image = cv2.imread('logs/2024-12-17T15-00-23_vs_random_cond_simple/images/train/reconstructions_gs-003500_e-000013_b-000486.png')
sam.segment_img(image)