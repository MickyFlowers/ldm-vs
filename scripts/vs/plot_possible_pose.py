import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def plot_bounding_box(ax, points, color, alpha=0.1):
    """
    更稳健的包围盒绘制方法
    """
    if len(points) < 2:
        return
    
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # 确保边界有效
    if np.any(max_vals - min_vals < 1e-6):
        print("Warning: Degenerate bounding box dimensions")
        return
    
    # 创建包围盒的8个顶点
    vertices = np.array([
        [min_vals[0], min_vals[1], min_vals[2]],
        [min_vals[0], min_vals[1], max_vals[2]],
        [min_vals[0], max_vals[1], min_vals[2]],
        [min_vals[0], max_vals[1], max_vals[2]],
        [max_vals[0], min_vals[1], min_vals[2]],
        [max_vals[0], min_vals[1], max_vals[2]],
        [max_vals[0], max_vals[1], min_vals[2]],
        [max_vals[0], max_vals[1], max_vals[2]]
    ])
    
    # 定义包围盒的12条边
    edges = [
        [vertices[0], vertices[1]],
        [vertices[0], vertices[2]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[3]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[3]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]],
        [vertices[4], vertices[5]],
        [vertices[4], vertices[6]],
        [vertices[5], vertices[7]],
        [vertices[6], vertices[7]]
    ]
    
    # 绘制边而不是面
    for edge in edges:
        edge = np.array(edge)
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=color, alpha=alpha, linewidth=1)

def visualize_workspace(left_arm_poses, right_arm_poses, object_poses, niuniu_poses, filename="workspace_visualization.png"):
    """
    可视化工作空间并保存为图片
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置颜色和标签
    colors = {
        'left_arm': 'r',
        'right_arm': 'b',
        'object': 'g',
        'niuniu': 'm'
    }
    
    # 提取位置坐标
    def extract_positions(poses):
        return [pose[:3, 3] for pose in poses]
    
    # 提取点集
    if left_arm_poses:
        left_positions = np.array(extract_positions(left_arm_poses))
        ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], 
                   c=colors['left_arm'], s=40, alpha=0.7, label='Left Arm TCP')
        plot_bounding_box(ax, left_positions, colors['left_arm'])
    
    if right_arm_poses:
        right_positions = np.array(extract_positions(right_arm_poses))
        ax.scatter(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], 
                   c=colors['right_arm'], s=30, alpha=0.5, label='Right Arm TCP')
        plot_bounding_box(ax, right_positions, colors['right_arm'])
    
    if object_poses:
        object_positions = np.array(extract_positions(object_poses))
        ax.scatter(object_positions[:, 0], object_positions[:, 1], object_positions[:, 2], 
                   c=colors['object'], s=25, alpha=0.6, label='Object')
    
    if niuniu_poses:
        niuniu_positions = np.array(extract_positions(niuniu_poses))
        ax.scatter(niuniu_positions[:, 0], niuniu_positions[:, 1], niuniu_positions[:, 2], 
                   c=colors['niuniu'], s=15, alpha=0.8, label='Niuniu')
    
    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Workspace Analysis')
    ax.legend()
    
    # 自动调整视图范围
    all_positions = []
    if left_arm_poses: all_positions.extend(extract_positions(left_arm_poses))
    if right_arm_poses: all_positions.extend(extract_positions(right_arm_poses))
    if object_poses: all_positions.extend(extract_positions(object_poses))
    if niuniu_poses: all_positions.extend(extract_positions(niuniu_poses))
    
    if all_positions:
        all_positions = np.array(all_positions)
        min_vals = np.min(all_positions, axis=0)
        max_vals = np.max(all_positions, axis=0)
        
        margin = 0.1
        ax.set_xlim(min_vals[0]-margin, max_vals[0]+margin)
        ax.set_ylim(min_vals[1]-margin, max_vals[1]+margin)
        ax.set_zlim(min_vals[2]-margin, max_vals[2]+margin)
    
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Visualization saved to {filename}")
    plt.close()

# 主程序
def main():
    # 初始化数据存储
    left_arm_poses = []
    right_arm_poses = []
    object_poses = []
    niuniu_poses = []
    
    # 初始位姿
    left_arm_pose = np.eye(4)
    left_arm_pose[:3, 3] = np.array([1.28588757e-01, 7.05102847e-01, 3.89579847e-01])
    left_arm_pose[:3, :3] =[[-0.9973036  , 0.05992138 ,-0.04236704],
                        [ 0.06389898 , 0.99295556 ,-0.09978064],
                        [ 0.0360896  ,-0.1022188 , -0.99410707]]
    left_arm_poses.append(left_arm_pose.copy())
    
    # 变换矩阵
    tip_to_tcp = np.eye(4)
    tip_to_tcp[2, 3] = 0.275
    
    niuniu_to_tip = np.eye(4)
    niuniu_to_tip[2, 3] = 0.15
    
    start_object_pose = left_arm_pose @ tip_to_tcp
    object_poses.append(start_object_pose.copy())
    
    start_niuniu_pose = start_object_pose @ niuniu_to_tip
    niuniu_poses.append(start_niuniu_pose.copy())
    
    # 误差范围
    in_hand_error_pos_lower = [-0.01, -0.02, -0.03]
    in_hand_error_pos_upper = [0.01, 0.0, 0.02]
    in_hand_error_rot_lower = [-45, -10, -20]
    in_hand_error_rot_upper = [-20, 10, 20]
    
    left_arm_error_pos_lower = np.array([-0.005, -0.005, 0.001])
    left_arm_error_pos_upper = np.array([0.005, 0.005, -0.001])
    left_arm_error_rot_lower = np.array([-5, -5, -5])
    left_arm_error_rot_upper = np.array([5, 5, 5])
    
    # 外循环（2次）
    for i in range(2):
        # 生成左臂误差
        left_arm_error_pos = np.random.uniform(
            left_arm_error_pos_lower, left_arm_error_pos_upper
        )
        left_arm_error_rot = np.random.uniform(
            left_arm_error_rot_lower, left_arm_error_rot_upper
        )
        
        # 构建误差变换矩阵
        left_arm_error_trans_matrix = np.eye(4)
        left_arm_error_trans_matrix[:3, :3] = R.from_euler(
            "XYZ", left_arm_error_rot, degrees=True
        ).as_matrix()
        left_arm_error_trans_matrix[:3, 3] = left_arm_error_pos
        
        # 计算新的位姿
        cur_niuniu_pose = start_niuniu_pose @ left_arm_error_trans_matrix
        niuniu_poses.append(cur_niuniu_pose.copy())
        
        cur_object_pose = cur_niuniu_pose @ np.linalg.inv(niuniu_to_tip)
        object_poses.append(cur_object_pose.copy())
        
        cur_left_arm_pose = cur_object_pose @ np.linalg.inv(tip_to_tcp)
        left_arm_poses.append(cur_left_arm_pose.copy())
        
        # 内循环（20次）
        for j in range(20):
            # 生成手部误差
            in_hand_error_pos = np.random.uniform(
                in_hand_error_pos_lower, in_hand_error_pos_upper
            )
            in_hand_error_rot = np.random.uniform(
                in_hand_error_rot_lower, in_hand_error_rot_upper
            )
            
            # 构建手部误差矩阵
            in_hand_error_trans_matrix = np.eye(4)
            in_hand_error_trans_matrix[:3, :3] = R.from_euler(
                "XYZ", in_hand_error_rot, degrees=True
            ).as_matrix()
            in_hand_error_trans_matrix[:3, 3] = in_hand_error_pos
            
            # 计算位姿
            tip_pose = cur_object_pose @ in_hand_error_trans_matrix
            tcp_pose = tip_pose @ np.linalg.inv(tip_to_tcp)
            
            # 模拟相机位姿变换
            left_camera_to_tcp = np.eye(4)
            right_camera_to_tcp = np.eye(4)
            
            camera_pose = tcp_pose @ left_camera_to_tcp
            right_tcp_pose = camera_pose @ np.linalg.inv(right_camera_to_tcp)
            
            right_arm_poses.append(right_tcp_pose.copy())
    
    # 输出结果
    print("="*50)
    print("Workspace Analysis Results")
    print("="*50)
    print(f"Left Arm Positions Generated: {len(left_arm_poses)}")
    print(f"Right Arm Positions Generated: {len(right_arm_poses)}")
    print(f"Object Positions Generated: {len(object_poses)}")
    print(f"Niuniu Positions Generated: {len(niuniu_poses)}")
    
    # 可视化并保存
    visualize_workspace(left_arm_poses, right_arm_poses, object_poses, niuniu_poses)

if __name__ == "__main__":
    main()