import os
import cv2


# rename img images

# root_path = "/mnt/workspace/cyxovo/dataset/20250912_charge_m/img"
# new_root_path = "/mnt/workspace/cyxovo/dataset/20250912_charge_m/rename_img"
# os.makedirs(new_root_path, exist_ok=True)
# files = os.listdir(root_path)
# files.sort(
#     key=lambda x: int(os.path.splitext(x)[0].split("-")[0]) * 30
#     + int(os.path.splitext(x)[0].split("-")[1])
# )

# for i, file in enumerate(files):
#     count = int(os.path.splitext(file)[0].split("-")[0]) * 30 + int(
#         os.path.splitext(file)[0].split("-")[1]
#     )
#     img = cv2.imread(os.path.join(root_path, file))
#     cv2.imwrite(os.path.join(new_root_path, f"{count:05d}.jpg"), img)

# # rename seg images
root_path = "/mnt/workspace/cyxovo/dataset/20250912_charge_m/rename_seg"
new_root_path = "/mnt/workspace/cyxovo/dataset/20250912_charge_m/seg"
os.makedirs(new_root_path, exist_ok=True)
files = os.listdir(root_path)
files.sort(key=lambda x: int(os.path.splitext(x)[0]))
for i, file in enumerate(files):
    count1 = int(os.path.splitext(file)[0]) // 30
    count2 = int(os.path.splitext(file)[0]) % 30
    img = cv2.imread(os.path.join(root_path, file))
    cv2.imwrite(os.path.join(new_root_path, f"{count1:03d}-{count2:03d}.jpg"), img)


# # import random
# # a = range(70)
# # selected_numbers = random.sample(a, 70)
# # selected_numbers.sort()
# # num_per_pose =
# # for i in range(1392):
# #     image1 = cv2.imread(f"../data/good_random_data_single_2/img/{i:05d}.jpg")
# #     image2 = cv2.imread(f"../data/good_random_data_single_2/ref/{i:05d}.jpg")
# #     cv2.imwrite(f"../data/good_random_data_single_2/img1/{i:03d}-{0:03d}.jpg", image1)
# #     cv2.imwrite(f"../data/good_random_data_single_2/img1/{i:03d}-{1:03d}.jpg", image2)
