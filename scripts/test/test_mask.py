import cv2
import numpy as np

image = cv2.imread("data/inpainting_examples/bench2.png")
mask = cv2.imread("data/inpainting_examples/bench2_mask.png", 0)

mask = mask.astype(np.float32) / 255.0
mask = mask[..., None]
mask[mask < 0.5] = 0
mask[mask >= 0.5] = 1

masked_image = (1 - mask) * image
cv2.imshow(":", masked_image.astype(np.uint8))
cv2.waitKey(0)
