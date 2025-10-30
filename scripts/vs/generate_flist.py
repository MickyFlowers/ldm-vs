import os
import random

root_path = "/mnt/workspace/cyxovo/dataset/screwdriver_sim"

flist = []
a = range(5000)
selected_numbers = random.sample(a, 500)
remaining_numbers = list(set(a) - set(selected_numbers))
selected_numbers.sort()
remaining_numbers.sort()

print(selected_numbers)

with open(os.path.join(root_path, "train_vs.flist"), "w") as f:
    for i in remaining_numbers:
        img = f"{i:05d}"
        f.write(img + "\n")

with open(os.path.join(root_path, "val_vs.flist"), "w") as f:
    for i in selected_numbers:
        img = f"{i:05d}"
        f.write(img + "\n")


# root_path = "/mnt/workspace/cyxovo/dataset"
# flist = []
# a = range(175000)
# selected_numbers = random.sample(a, 3000)
# selected_numbers_val = random.sample(selected_numbers, 300)
# selected_numbers_train = list(set(selected_numbers) - set(selected_numbers_val))
# selected_numbers_val.sort()
# selected_numbers_train.sort()

# b = range(1392 * 4)
# selected_numbers_val_2 = random.sample(b, 500)
# selected_numbers_train_2 = list(set(b) - set(selected_numbers_val_2))
# selected_numbers_val_2.sort()
# selected_numbers_train_2.sort()

# c = range(10000)
# selected_numbers_val_3 = random.sample(c, 1000)
# selected_numbers_train_3 = list(set(c) - set(selected_numbers_val_3))
# selected_numbers_val_3.sort()
# selected_numbers_train_2.sort()

# with open(os.path.join(root_path, "train_vs.flist"), "w") as f:
#     for i in selected_numbers_train:
#         count1 = i // 2500
#         count2 = (i % 2500) % 50
#         count3 = (i % 2500) // 50
#         img = f"good_random_data_single,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
#         f.write(img + "\n")
#     for i in selected_numbers_train_2:
#         count1 = i // 4
#         count2 = (i % 4) % 2
#         count3 = (i % 4) // 2
#         img = f"good_random_data_single_2,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
#         f.write(img + "\n")
#     for i in selected_numbers_train_3:
#         count1 = i // 25
#         count2 = (i % 25) % 5
#         count3 = (i % 25) // 5
#         img = f"good_random_data_single_3,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
#         f.write(img + "\n")

# with open(os.path.join(root_path, "val_vs.flist"), "w") as f:
#     for i in selected_numbers_val:
#         count1 = i // 2500
#         count2 = (i % 2500) % 50
#         count3 = (i % 2500) // 50
#         img = f"good_random_data_single,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
#         f.write(img + "\n")
#     for i in selected_numbers_val_2:
#         count1 = i // 4
#         count2 = (i % 4) % 2
#         count3 = (i % 4) // 2
#         img = f"good_random_data_single_2,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
#         f.write(img + "\n")
#     for i in selected_numbers_val_3:
#         count1 = i // 25
#         count2 = (i % 25) % 5
#         count3 = (i % 25) // 5
#         img = f"good_random_data_single_3,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
#         f.write(img + "\n")
