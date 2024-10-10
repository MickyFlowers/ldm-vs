import os
import random
root_path = "/yuxi/data/vs/real_world_data_4"

flist = []
a = range(11700)
selected_numbers = random.sample(a, 1700)
remaining_numbers = list(set(a) - set(selected_numbers))
selected_numbers.sort()
remaining_numbers.sort()

print(selected_numbers)

with open(os.path.join(root_path, "train.flist"), "w") as f:
    for i in remaining_numbers:
        img = f"{i:05d}"
        f.write(img + ".jpg" + "\n")

with open(os.path.join(root_path, "val.flist"), "w") as f:
    for i in selected_numbers:
        img = f"{i:05d}"
        f.write(img + ".jpg" + "\n")