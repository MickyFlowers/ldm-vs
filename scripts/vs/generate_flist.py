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


root_path = "/home/cyx/project/data"
flist = []
a = range(175000)
selected_numbers = random.sample(a, 3000)
selected_numbers_val = random.sample(selected_numbers, 300)
selected_numbers_train = list(set(selected_numbers) - set(selected_numbers_val))
selected_numbers_val.sort()
selected_numbers_train.sort()

b = range(1392 * 2)
selected_numbers_val_2 = random.sample(b, 784)
selected_numbers_train_2 = list(set(b) - set(selected_numbers_val_2))
selected_numbers_val_2.sort()
selected_numbers_train_2.sort()

with open(os.path.join(root_path, "train_vs.flist"), "w") as f:
    for i in selected_numbers_train:
        count1 = i // 2500
        count2 = (i % 2500) % 50
        count3 = (i % 2500) // 50
        img = f"good_random_data_single,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
    for i in selected_numbers_train_2:
        count1 = i // 4
        count2 = (i % 4) % 2
        count3 = (i % 4) // 2
        img = f"good_random_data_single_2,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
        
with open(os.path.join(root_path, "val_vs.flist"), "w") as f:
    for i in selected_numbers_val:
        count1 = i // 2500
        count2 = (i % 2500) % 50
        count3 = (i % 2500) // 50
        img = f"good_random_data_single,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
    for i in selected_numbers_val_2:
        count1 = i // 4
        count2 = (i % 4) % 2
        count3 = (i % 4) // 2
        img = f"good_random_data_single_2,img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
