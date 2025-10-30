import os
import random
from tqdm import tqdm
# root_path = "/mnt/pfs/asdfe1/data/2025-09-02-screwdriver-sim-m"

# flist = []
# a = range(5000)
# selected_numbers = random.sample(a, 500)
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/mnt/workspace/cyxovo/dataset/20250912_charge_m")
parser.add_argument("--num_pose", type=int, default=100)
parser.add_argument("--num_camera_pose_per_pose", type=int, default=30)
args = parser.parse_args()
# flist = []
a = range(args.num_pose * args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)
# selected_numbers = random.sample(a, int(args.num_pose * args.num_camera_pose_per_pose * args.num_camera_pose_per_pose * 0.1))
selected_numbers_val = random.sample(a, int(args.num_pose * args.num_camera_pose_per_pose * args.num_camera_pose_per_pose * 0.1))
selected_numbers_train = list(set(a) - set(selected_numbers_val))
selected_numbers_val.sort()
selected_numbers_train.sort()

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

with open(os.path.join(args.root_path, "train_ldm.flist"), "w") as f:
    for i in selected_numbers_train:
        count1 = i // (args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)
        count2 = (i % (args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)) % args.num_camera_pose_per_pose
        count3 = (i % (args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)) // args.num_camera_pose_per_pose
        img = f"img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
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

with open(os.path.join(args.root_path, "val_ldm.flist"), "w") as f:
    for i in selected_numbers_val:
        count1 = i // (args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)
        count2 = (i % (args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)) % args.num_camera_pose_per_pose
        count3 = (i % (args.num_camera_pose_per_pose * args.num_camera_pose_per_pose)) // args.num_camera_pose_per_pose
        img = f"img/{count1:03d}-{count2:03d}.jpg,img/{count1:03d}-{count3:03d}.jpg,seg/{count1:03d}-{count2:03d}.jpg"
        f.write(img + "\n")
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



# #data_total_flist
# root_path = "/mnt/workspace/cyxovo/dataset/charge_data_random"
# flist = []
# a = range(12000)
# selected_numbers_train = list(set(a))


# with open(os.path.join(root_path, "total_charge.flist"), "w") as f:
#    for i in tqdm(selected_numbers_train, desc="Processing training set"):  # 添加进度条
#         count1 = i // 10
#         count2 = i % 10
#         for count3 in range(10):
#             img = f"img/{count1:05d}-{count2:05d}.jpg,img/{count1:05d}-{count3:05d}.jpg,seg/{count1:05d}-{count2:05d}.jpg"
#             f.write(img + "\n")


#random_data_flist

# def sample_flist(input_file, train_file, test_file, sample_size=36000, total_lines=120000):
#     # 生成要采样的行号（从0开始）
#     sampled_indices = random.sample(range(total_lines), sample_size)
#     sampled_indices = set(sampled_indices)  # 转换为集合加快查找速度
    
#     with open(input_file, 'r') as f_in, open(train_file, 'w') as train_out, open(test_file, 'w') as test_out:
#         for i, line in enumerate(f_in):
#             if i in sampled_indices:
#                 test_out.write(line)
#             else:
#                 train_out.write(line)

# if __name__ == '__main__':
   
#     input_flist = '/mnt/workspace/cyxovo/dataset/charge_data_random/total_charge.flist'  
    
#     train_flist = '/mnt/workspace/cyxovo/dataset/charge_data_random/train_random.flist'  
#     test_flist = '/mnt/workspace/cyxovo/dataset/charge_data_random/test_random.flist'  
    
    
#     sample_flist(input_flist, train_flist, test_flist)
#     print(f"已从 {input_flist} 中随机选取36000行写入 {test_flist},剩下的写入{train_flist}")

#single_data_flist
# root_path = "/mnt/pfs/asdfe1/data/2025-09-02-screwdriver-sim-m"
# flist = []
# a = range(18000)
# val_numbers = random.sample(a, 1800)
# train_numbers = list(set(a) - set(val_numbers))
# val_numbers.sort()
# train_numbers.sort()


# train_file = os.path.join(root_path, "train_ldm.flist")
# test_file = os.path.join(root_path, "val_ldm.flist")

# with open(train_file, 'w') as train_out, open(test_file, 'w') as test_out:
#     for i in train_numbers:
#         count1 = i // 30
#         count2 = i % 30
#         train_out.write(f"img/{count1:05d}_{count2:05d}.jpg\n")
#     for i in val_numbers:
#         count1 = i // 10
#         count2 = i % 10
#         test_out.write(f"img/{count1:05d}_{count2:05d}.jpg\n")
