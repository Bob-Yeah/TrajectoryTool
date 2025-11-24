import re
import TranslatedFunctions
import HandPR
def extract_vector_data(line):
    """
    从一行日志中提取Vector3数据（三个浮点数）
    例如：从 "[2025-10-15 16:22:14] 手部原始轨迹信息:(0.5339, 1.4460, 0.2466)" 中提取 (0.5339, 1.4460, 0.2466)
    """
    pattern = r'\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)'
    match = re.search(pattern, line)
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            return (x, y, z)
        except ValueError:
            return None
    return None

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# === 关键配置：在绘图代码之前设置 ===
# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 常用
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 常用
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux 常用

# 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# === 配置结束 ===

def visualize_trajectories_3d(all_sections_data, normals):
    """
    对每组轨迹点进行3D可视化，两个一组显示在同一个窗口中，标记起点和终点，并显示法向量
    
    参数:
        all_sections_data: 列表的列表，每个子列表包含该部分的轨迹点元组(x, y, z)
        normals: 列表，每个元素是对应数据段的法向量元组(x, y, z)
    """
    # 检查是否有数据可可视化
    total_sections = len(all_sections_data)
    valid_sections = sum(1 for section in all_sections_data if len(section) > 0)
    
    print(f"总共 {total_sections} 个数据部分，其中 {valid_sections} 个部分有轨迹数据")
    
    if valid_sections == 0:
        print("没有找到有效的轨迹数据可可视化")
        return
    
    # 确保法向量列表与数据段数量匹配
    if normals and len(normals) < total_sections:
        print(f"警告：法向量数量({len(normals)})少于数据段数量({total_sections})")
    
    # 将数据两个一组处理
    i = 0
    group_count = 1
    while i < total_sections:
        # 获取当前组的两个数据段
        group_sections = []
        group_indices = []
        group_normals = []
        
        # 收集最多两个有效的数据段
        for j in range(i, min(i + 2, total_sections)):
            if len(all_sections_data[j]) > 0:
                group_sections.append(all_sections_data[j])
                group_indices.append(j + 1)  # +1是因为部分编号从1开始
                # 收集对应的法向量
                if normals and j < len(normals):
                    group_normals.append(normals[j])
                else:
                    group_normals.append((0, 0, 1))  # 默认向上法向量
        
        # 如果当前组有数据，进行可视化
        if group_sections:
            if len(group_sections) == 1:
                # 如果只有一个数据段，使用原来的函数并传递法向量
                visualize_single_section(group_sections[0], group_indices[0], 
                                        group_normals[0] if group_normals else (0, 0, 1))
            else:
                # 如果有两个数据段，使用新函数在同一窗口显示并传递法向量
                visualize_two_sections(group_sections, group_indices, group_count, group_normals)
        
        # 移动到下一组
        i += 2
        group_count += 1
    
    # 所有图形创建完成后，一次性显示所有窗口
    print("所有轨迹可视化窗口已创建，请关闭所有窗口来结束程序")
    plt.show()  # 这会显示所有创建的图形窗口

def visualize_two_sections(sections_data, section_indices, group_num, normals=None):
    """
    在同一个窗口中可视化两个数据段的轨迹，并显示每个数据段的法向量
    
    参数:
        sections_data: 包含两个数据段的列表，每个数据段是轨迹点元组的列表
        section_indices: 两个数据段的原始编号
        group_num: 分组编号
        normals: 包含两个数据段法向量的列表，每个法向量是(x, y, z)元组
    """
    # 如果未提供法向量，使用默认值
    if normals is None:
        normals = [(0, 0, 1), (0, 0, 1)]  # 默认向上法向量
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置图形标题
    ax.set_title(f'轨迹组 {group_num} - 数据段 {section_indices[0]} 和 {section_indices[1]}', 
                 fontsize=14, pad=20)
    
    # 使用的颜色映射
    colormaps = [cm.viridis, cm.plasma]  # 为两个数据段使用不同的颜色映射
    
    # 处理每个数据段
    for segment_idx, (section_data, original_section_num) in enumerate(zip(sections_data, section_indices)):
        # 提取坐标数据
        x_vals = [point[0] for point in section_data]
        y_vals = [point[1] for point in section_data]
        z_vals = [point[2] for point in section_data]
        
        # 创建颜色映射，从起始到结束表示轨迹方向
        colors = np.linspace(0, 1, len(section_data))
        colormap = colormaps[segment_idx]
        
        # 计算并设置合适的坐标范围，确保所有数据可见
        all_x = []
        all_y = []
        all_z = []
        for section_data in sections_data:
            all_x.extend([point[0] for point in section_data])
            all_y.extend([point[1] for point in section_data])
            all_z.extend([point[2] for point in section_data])
        
        max_range = max(max(all_x) - min(all_x), 
                    max(all_y) - min(all_y), 
                    max(all_z) - min(all_z)) * 0.5
                    
        # 绘制轨迹线
        for j in range(len(section_data) - 1):
            ax.plot([x_vals[j], x_vals[j+1]], 
                    [y_vals[j], y_vals[j+1]], 
                    [z_vals[j], z_vals[j+1]], 
                    color=colormap(colors[j]), 
                    linewidth=2, 
                    alpha=0.8, 
                    label=f'数据段 {original_section_num}' if j == 0 else "")
        
        # 标记所有轨迹点
        scatter = ax.scatter(x_vals, y_vals, z_vals, 
                           c=colors, cmap=colormap, 
                           s=20, alpha=0.6, 
                           label=f'数据段 {original_section_num} 轨迹点' if segment_idx == 0 else "")
        
        # 计算重心
        mean_x = np.mean(x_vals)
        mean_y = np.mean(y_vals)
        mean_z = np.mean(z_vals)
        
        # 标记重心点（使用不同颜色区分不同数据段）
        centroid_colors = ['blue', 'purple']
        ax.scatter([mean_x], [mean_y], [mean_z], 
                  color=centroid_colors[segment_idx], marker='o', s=150, 
                  label=f'数据段 {original_section_num} 重心', edgecolors='black', linewidth=1)
        
        # 特别标记起点和终点
        # 起点 - 使用不同颜色区分不同数据段
        start_colors = ['green', 'limegreen']
        ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], 
                  color=start_colors[segment_idx], marker='o', s=150, 
                  label=f'数据段 {original_section_num} 起点' if segment_idx == 0 else "", 
                  edgecolors='black', linewidth=1)
        
        # 在终点绘制带箭头的法向量
        # 获取当前数据段的法向量
        if segment_idx < len(normals):
            normal = normals[segment_idx]
        else:
            normal = (0, 0, 1)  # 默认向上
        
        # 计算法向量的比例因子
        scale_factor = max_range * 0.3  # 使用最大范围的30%作为法向量长度
        
        # 规范化法向量
        normal_mag = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if normal_mag > 1e-6:  # 避免除以零
            scaled_normal = (normal[0] * scale_factor / normal_mag, 
                             normal[1] * scale_factor / normal_mag, 
                             normal[2] * scale_factor / normal_mag)
        else:
            scaled_normal = (0, 0, scale_factor)
        
        # 使用不同颜色区分两个数据段的法向量
        normal_colors = ['orange', 'pink']
        
        # 在终点绘制法向量
        ax.quiver(mean_x, mean_y, mean_z,  # 起点
                  scaled_normal[0], scaled_normal[1], scaled_normal[2],  # 向量
                  color=normal_colors[segment_idx], arrow_length_ratio=0.3, linewidth=3,
                  label=f'数据段 {original_section_num} 法向量' if segment_idx == 0 else "")
        
        if (segment_idx == 1):
            # 绘制半透明平面 - 由法向量决定朝向，由法向量起点决定位置
            # 确保法向量已经规范化
            normal_vector = np.array(scaled_normal) / max(normal_mag, 1e-6)
            
            # 创建平面上的两个正交向量，用于生成网格
            # 使用叉乘找到与法向量垂直的向量
            if abs(normal_vector[0]) < abs(normal_vector[1]):
                u = np.array([1, 0, 0])
            else:
                u = np.array([0, 1, 0])
            u = np.cross(u, normal_vector)
            u = u / np.linalg.norm(u)
            v = np.cross(normal_vector, u)
            v = v / np.linalg.norm(v)
            
            # 平面方程: ax + by + cz + d = 0
            a, b, c = normal_vector
            d = -(a * mean_x + b * mean_y + c * mean_z)
            
            # 创建网格点
            size = scale_factor * 2  # 平面大小与法向量长度相关
            x_grid, y_grid = np.meshgrid(np.linspace(mean_x - size, mean_x + size, 20),
                                        np.linspace(mean_y - size, mean_y + size, 20))
            
            # 计算z坐标（如果法向量的z分量不为零）
            if abs(c) > 1e-6:
                z_grid = (-a * x_grid - b * y_grid - d) / c
            else:
                # 如果法向量在z方向分量很小，使用y坐标计算
                if abs(b) > 1e-6:
                    z_grid = np.zeros_like(x_grid)
                    y_grid = (-a * x_grid - c * z_grid - d) / b
                else:
                    # 如果法向量在y方向分量也很小，使用x坐标计算
                    z_grid = np.zeros_like(y_grid)
                    x_grid = (-b * y_grid - c * z_grid - d) / a
            
            # 绘制半透明平面

            plane_color = normal_colors[segment_idx]
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color=plane_color,
                        edgecolor='none', label=f'数据段 {original_section_num} 平面' if segment_idx == 0 else "")
            
        # 终点 - 使用不同颜色区分不同数据段
        end_colors = ['red', 'orange']
        ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 
                  color=end_colors[segment_idx], marker='X', s=150, 
                  label=f'数据段 {original_section_num} 终点' if segment_idx == 0 else "", 
                  edgecolors='black', linewidth=1)
        
        # 添加从起点到终点的直线（虚线）
        ax.plot([x_vals[0], x_vals[-1]], 
                [y_vals[0], y_vals[-1]], 
                [z_vals[0], z_vals[-1]], 
                '--', alpha=0.5, linewidth=1, 
                color=end_colors[segment_idx])
    
    # 设置图形属性
    ax.set_xlabel('X 轴', fontsize=12, labelpad=10)
    ax.set_ylabel('Y 轴', fontsize=12, labelpad=10)
    ax.set_zlabel('Z 轴', fontsize=12, labelpad=10)
    
    
    
    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 调整视角以便更好地观察
    ax.view_init(elev=20, azim=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例，并确保不重复显示相同标签
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
              bbox_to_anchor=(0, 1), fontsize=10)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 打印轨迹信息
    for i, (section_data, section_idx) in enumerate(zip(sections_data, section_indices)):
        print(f"组 {group_num} - 数据段 {section_idx}: {len(section_data)} 个轨迹点")
        print(f"  起点坐标: ({section_data[0][0]:.3f}, {section_data[0][1]:.3f}, {section_data[0][2]:.3f})")
        print(f"  终点坐标: ({section_data[-1][0]:.3f}, {section_data[-1][1]:.3f}, {section_data[-1][2]:.3f})")

def visualize_single_section(section_data, section_num, normal=(0, 0, 1)):
    """
    可视化单个部分的轨迹数据，包括在终点绘制法向量
    
    参数:
        section_data: 该部分的轨迹点列表，每个元素为(x, y, z)元组
        section_num: 部分编号
        normal: 法向量元组(x, y, z)，默认为向上
    """
    # 提取坐标数据
    x_vals = [point[0] for point in section_data]
    y_vals = [point[1] for point in section_data]
    z_vals = [point[2] for point in section_data]
    
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建颜色映射，从蓝色到红色表示轨迹方向
    colors = np.linspace(0, 1, len(section_data))
    colormap = cm.viridis
    
    # 绘制轨迹线
    for j in range(len(section_data) - 1):
        ax.plot([x_vals[j], x_vals[j+1]], 
                [y_vals[j], y_vals[j+1]], 
                [z_vals[j], z_vals[j+1]], 
                color=colormap(colors[j]), 
                linewidth=2, 
                alpha=0.8)
    
    # 标记所有轨迹点（较小点）
    scatter = ax.scatter(x_vals, y_vals, z_vals, 
                       c=colors, cmap='viridis', 
                       s=20, alpha=0.6, 
                       label='轨迹点')
    
    mean_x = np.mean(x_vals)
    mean_y = np.mean(y_vals)
    mean_z = np.mean(z_vals)


    ax.scatter([mean_x], [mean_y], [mean_z], 
              color='blue', marker='o', s=200, 
              label='重心点', edgecolors='black', linewidth=2)
    
    # 特别标记起点和终点
    # 起点 - 绿色大圆点
    ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], 
              color='green', marker='o', s=200, 
              label='起点', edgecolors='black', linewidth=2)
    
    # 终点 - 红色大叉号
    ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 
              color='red', marker='X', s=200, 
              label='终点', edgecolors='black', linewidth=2)
    
    # 设置等比例坐标轴，保持比例正确
    max_range = max(max(x_vals)-min(x_vals), 
                   max(y_vals)-min(y_vals), 
                   max(z_vals)-min(z_vals)) * 0.5

    # 计算法向量的比例因子，使其长度与图形尺寸相匹配
    scale_factor = max_range * 0.3  # 使用最大范围的30%作为法向量长度
    
    # 规范化法向量
    normal_mag = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if normal_mag > 1e-6:  # 避免除以零
        scaled_normal = (normal[0] * scale_factor / normal_mag, 
                         normal[1] * scale_factor / normal_mag, 
                         normal[2] * scale_factor / normal_mag)
    else:
        scaled_normal = (0, 0, scale_factor)  # 默认向上
    
    # 在终点绘制带箭头的法向量
    ax.quiver(mean_x, mean_y, mean_z,  # 起点
              scaled_normal[0], scaled_normal[1], scaled_normal[2],  # 向量
              color='orange', arrow_length_ratio=0.3, linewidth=3, 
              label='法向量')
    
    # 绘制半透明平面 - 由法向量决定朝向，由法向量起点决定位置
    # 确保法向量已经规范化
    normal_vector = np.array(scaled_normal) / max(normal_mag, 1e-6)
    
    # 创建平面上的两个正交向量，用于生成网格
    # 使用叉乘找到与法向量垂直的向量
    if abs(normal_vector[0]) < abs(normal_vector[1]):
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    u = np.cross(u, normal_vector)
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vector, u)
    v = v / np.linalg.norm(v)
    
    # 平面方程: ax + by + cz + d = 0
    a, b, c = normal_vector
    d = -(a * mean_x + b * mean_y + c * mean_z)
    
    # 创建网格点
    size = scale_factor * 2  # 平面大小与法向量长度相关
    x_grid, y_grid = np.meshgrid(np.linspace(mean_x - size, mean_x + size, 20),
                                 np.linspace(mean_y - size, mean_y + size, 20))
    
    # 计算z坐标（如果法向量的z分量不为零）
    if abs(c) > 1e-6:
        z_grid = (-a * x_grid - b * y_grid - d) / c
    else:
        # 如果法向量在z方向分量很小，使用y坐标计算
        if abs(b) > 1e-6:
            z_grid = np.zeros_like(x_grid)
            y_grid = (-a * x_grid - c * z_grid - d) / b
        else:
            # 如果法向量在y方向分量也很小，使用x坐标计算
            z_grid = np.zeros_like(y_grid)
            x_grid = (-b * y_grid - c * z_grid - d) / a
    
    # 绘制半透明平面
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='orange',
                   edgecolor='none', label='法向量平面')
    
    # 添加从起点到终点的直线（虚线）
    ax.plot([x_vals[0], x_vals[-1]], 
            [y_vals[0], y_vals[-1]], 
            [z_vals[0], z_vals[-1]], 
            'k--', alpha=0.5, linewidth=1, label='起点-终点连线')
    
    # 设置图形属性
    ax.set_xlabel('X 轴', fontsize=12, labelpad=10)
    ax.set_ylabel('Y 轴', fontsize=12, labelpad=10)
    ax.set_zlabel('Z 轴', fontsize=12, labelpad=10)
    
    ax.set_title(f'手部轨迹可视化 - 部分 {section_num}\n'
                f'轨迹点数量: {len(section_data)}', 
                fontsize=14, pad=20)
    
    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('轨迹进度', rotation=270, labelpad=15)
    
    
    
    mid_x = (max(x_vals) + min(x_vals)) * 0.5
    mid_y = (max(y_vals) + min(y_vals)) * 0.5
    mid_z = (max(z_vals) + min(z_vals)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 调整视角以便更好地观察
    ax.view_init(elev=20, azim=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 只设置布局，不单独显示（会在所有图形创建完成后统一显示）
    plt.tight_layout()
    
    # 打印轨迹信息
    print(f"部分 {section_num}: {len(section_data)} 个轨迹点")
    print(f"起点坐标: ({x_vals[0]:.3f}, {y_vals[0]:.3f}, {z_vals[0]:.3f})")
    print(f"终点坐标: ({x_vals[-1]:.3f}, {y_vals[-1]:.3f}, {z_vals[-1]:.3f})")
    print(f"重心坐标: ({mean_x:.3f}, {mean_y:.3f}, {mean_z:.3f})")
    print(f"轨迹长度: {calculate_trajectory_length(section_data):.3f}\n")

def calculate_trajectory_length(section_data):
    """
    计算轨迹的总长度
    
    参数:
        section_data: 轨迹点列表
    
    返回:
        轨迹总长度
    """
    total_length = 0.0
    for i in range(len(section_data) - 1):
        point1 = section_data[i]
        point2 = section_data[i+1]
        distance = np.sqrt((point2[0]-point1[0])**2 + 
                          (point2[1]-point1[1])**2 + 
                          (point2[2]-point1[2])**2)
        total_length += distance
    return total_length

def create_comparison_plot(all_sections_data):
    """
    创建所有轨迹的对比图（可选功能）
    
    参数:
        all_sections_data: 所有部分的轨迹数据
    """
    valid_sections = [section for section in all_sections_data if len(section) > 0]
    
    if len(valid_sections) < 2:
        print("需要至少2个有效轨迹部分才能创建对比图")
        return
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_sections)))
    
    for i, section_data in enumerate(valid_sections):
        x_vals = [point[0] for point in section_data]
        y_vals = [point[1] for point in section_data]
        z_vals = [point[2] for point in section_data]
        
        # 绘制轨迹
        ax.plot(x_vals, y_vals, z_vals, 
               color=colors[i], linewidth=2, 
               label=f'轨迹 {i+1} ({len(section_data)}点)')
        
        # 标记起点和终点
        ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], 
                  color=colors[i], marker='o', s=100)
        ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 
                  color=colors[i], marker='s', s=100)
    
    ax.set_xlabel('X 轴')
    ax.set_ylabel('Y 轴')
    ax.set_zlabel('Z 轴')
    ax.set_title('多轨迹对比图', fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# part by part
def extract_data(lines):
    points = []
    points_clean = []
    puncture_norm = HandPR.Vector3(0, 0, 0)
    puncture_center = HandPR.Vector3(0, 0, 0)
    for line in lines:
        if 'Puncture:' in line:
            puncture_match = re.search(r'Puncture:\(([^)]+)\),\(([^)]+)\)', line)
            if puncture_match:
                # 提取两个坐标组
                vectors_str = puncture_match.groups()
                vec_str = vectors_str[0]
                # 分割坐标值并转换为浮点数
                coords = vec_str.split(',')
                if len(coords) == 3:
                    x = float(coords[0].strip())
                    y = float(coords[1].strip())
                    z = float(coords[2].strip())
                    puncture_norm = HandPR.Vector3(x, y, z)
                vec_str = vectors_str[1]
                # 分割坐标值并转换为浮点数
                coords = vec_str.split(',')
                if len(coords) == 3:
                    x = float(coords[0].strip())
                    y = float(coords[1].strip())
                    z = float(coords[2].strip())
                    puncture_center = HandPR.Vector3(x, y, z)
            print(f"puncture_norm: {puncture_norm}")
            print(f"puncture_center: {puncture_center}")

        if '手部原始轨迹信息' in line:
        # if '记录干净点' in line:
        # if '记录消毒原始轨迹信息' in line:
        # if '平面内点' in line:
        # if '降采样轨迹' in line:
        # if '蒙特卡洛散点' in line:
            vector_data = extract_vector_data(line)
            points.append(HandPR.Vector3(vector_data[0], vector_data[1], vector_data[2]))
    norm_cal = []
    if (len(points) > 0):
        # points_clean, norm_cal = HandPR.delete_outliers(points, puncture_norm, puncture_center)
        points_clean, norm_cal = HandPR.ransac_trajectory_cutting(points,100,0.01,0.03)
    return points, points_clean, puncture_norm, puncture_center, norm_cal

def main():
    # 请将文件名替换为您的实际log文件路径
    filename = '20251114_1605/UnityLog_160512_Hand.log'
    
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # 以“消毒环节手部追踪录制开始”为分隔符分割内容
    parts = content.split('录制开始')
    
    # 存储所有部分的数据，每个部分是一个包含Vector3元组的列表
    all_sections_data = []

    puncture_plane_norm = HandPR.Vector3(0,0,0)  # 示例法向量
    puncture_plane_center = HandPR.Vector3(0,0,0)  # 示例中心点
    normals = []
    for i, part in enumerate(parts):
        lines = part.split('\n')
        points, points_clean, puncture_plane_norm, puncture_plane_center, normal_cal = extract_data(lines)

        if (len(points_clean) > 0):
            # 轨迹评判
            _,_,_=TranslatedFunctions.clean_trajectory_check2(points_clean, normal_cal)
            
            normals.append([puncture_plane_norm.x, puncture_plane_norm.y, puncture_plane_norm.z])
            normals.append([normal_cal.x, normal_cal.y, normal_cal.z])

            all_sections_data.append(points)
            all_sections_data.append(points_clean)
        
    
    # 输出结果：每个部分的数据点数量
    for i, data in enumerate(all_sections_data):
        print(f"部分 {i} 有 {len(data)} 个轨迹数据点")
    
    # 如果您需要访问具体数据，可以使用 all_sections_data
    # 例如：第一个部分的第一个数据点 -> all_sections_data[0][0]
    
    # 可选：将数据保存到文件（JSON格式）
    import json
    # 将元组转换为列表，以便JSON序列化
    json_data = [[list(point) for point in section] for section in all_sections_data]
    # print(json_data)

    # with open('extracted_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(json_data, f, indent=2)
    # print("数据已保存到 extracted_data.json")

    # example_data = [
    #     [(0.1, 0.2, 0.3), (0.2, 0.3, 0.4), (0.3, 0.4, 0.5), (0.4, 0.5, 0.6)],  # 第一部分
    #     [],  # 空部分（将被跳过）
    #     [(0.5, 0.6, 0.7), (0.6, 0.7, 0.8), (0.7, 0.8, 0.9)]  # 第三部分
    # ]
    
    # 调用可视化函数
    visualize_trajectories_3d(json_data, normals)
    
    # 可选：创建对比图
    # create_comparison_plot(json_data)

def plot_float_trend(values, title='数值趋势', window=10, show_trend=True):
    """
    将一个 float 列表画出趋势线。
    - values: 浮点数列表
    - title: 图标题
    - window: 滑动平均窗口大小（None 或 <=1 表示不启用）
    - show_trend: 是否绘制线性趋势线
    """
    if not values:
        print('没有数值可绘制')
        return

    x = np.arange(len(values))
    plt.figure(figsize=(12, 5))

    # 原始序列
    plt.plot(x, values, label='原始序列', color='tab:blue', alpha=0.6)
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()