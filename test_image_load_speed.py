import cv2
import os
import time

root_path = "/mnt/pfs/asdfe1/data/250815-m/img/"
for i in range(1000):
    count1 = i // 20
    count2 = i % 20
    img_file_name = os.path.join(root_path, f"{count1:03d}-{count2:03d}.jpg".format(i))
    start_time = time.time()
    img = cv2.imread(img_file_name)
    print("--- %s seconds ---" % (time.time() - start_time))