import os
import random

# root_path = "/home/cyx/project/data"

# flist = []
# a = range(2857)
# selected_numbers = random.sample(a, 857)
# remaining_numbers = list(set(a) - set(selected_numbers))
# selected_numbers.sort()
# remaining_numbers.sort()

# print(selected_numbers)

# with open(os.path.join(root_path, "train_vs.flist"), "w") as f:
#     for i in remaining_numbers:
#         img = f"{i:05d}"
#         f.write(img + "\n")

# with open(os.path.join(root_path, "val_vs.flist"), "w") as f:
#     for i in selected_numbers:
#         img = f"{i:05d}"
#         f.write(img + "\n")


root_path = "/mnt/workspace/cyxovo/dataset"
flist = []
a = range(3500)
# selected_numbers = random.sample(a, 3000)
selected_numbers_val = random.sample(a, 500)
selected_numbers_train = list(set(a) - set(selected_numbers_val))
selected_numbers_val.sort()
selected_numbers_train.sort()

b = range(1392 * 2)
selected_numbers_val_2 = random.sample(b, 784)
selected_numbers_train_2 = list(set(b) - set(selected_numbers_val_2))
selected_numbers_val_2.sort()
selected_numbers_train_2.sort()

c = range(2000)
selected_numbers_val_3 = random.sample(c, 200)
selected_numbers_train_3 = list(set(c) - set(selected_numbers_val_3))
selected_numbers_val_3.sort()
selected_numbers_train_2.sort()

with open(os.path.join(root_path, "train_img.flist"), "w") as f:
    for i in selected_numbers_train:
        count1 = i // 50
        count2 = i % 50
        img = f"good_random_data_single,img/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
        
    for i in selected_numbers_train_2:
        count1 = i // 2
        count2 = i % 2
        img = f"good_random_data_single_2,img/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
    for i in selected_numbers_train_3:
        count1 = i // 5
        count2 = i % 5
        img = f"good_random_data_single_3,img/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")

with open(os.path.join(root_path, "val_img.flist"), "w") as f:
    for i in selected_numbers_val:
        count1 = i // 50
        count2 = i % 50
        img = f"good_random_data_single,img/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
    for i in selected_numbers_val_2:
        count1 = i // 2
        count2 = i % 2
        img = f"good_random_data_single_2,img/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
    for i in selected_numbers_val_3:
        count1 = i // 5
        count2 = i % 5
        img = f"good_random_data_single_3,img/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
