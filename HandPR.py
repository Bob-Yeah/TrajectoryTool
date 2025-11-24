from dataclasses import dataclass
from tracemalloc import start
from typing import List
import math

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def Vector3(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalized(self):
        mag = self.magnitude()
        return Vector3(self.x/mag, self.y/mag, self.z/mag)
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    
def calculate_mean(points):
    return Vector3(
        sum(p.x for p in points)/len(points),
        sum(p.y for p in points)/len(points),
        sum(p.z for p in points)/len(points)
    )

def calculate_plane_from_three_points(p1: Vector3, p2: Vector3, p3: Vector3) -> tuple:
    """
    通过三个点计算平面方程
    
    参数:
        p1, p2, p3: 三个Vector3点
    
    返回:
        (A, B, C, D, centroid): 平面方程系数和重心点
    """
    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p1
    
    # 计算法向量（叉乘）
    normal = v1.cross(v2)
    
    # 检查三个点是否共线
    if normal.magnitude() < 1e-8:
        raise ValueError("三点共线，无法确定唯一平面")
    
    # 归一化法向量
    normal = normal.normalized()
    
    # 平面方程: Ax + By + Cz + D = 0
    A, B, C = normal.x, normal.y, normal.z
    D = -normal.dot(p1)  # 使用p1计算D
    
    # 计算三个点的重心
    centroid = calculate_mean([p1, p2, p3])
    
    return A, B, C, D, centroid

def calculate_distance_to_plane(point: Vector3, plane_coeffs: tuple) -> float:
    """
    计算点到平面的距离
    
    参数:
        point: Vector3点
        plane_coeffs: (A, B, C, D)平面方程系数
    
    返回:
        点到平面的距离
    """
    A, B, C, D = plane_coeffs
    # 距离公式: |Ax + By + Cz + D| / sqrt(A² + B² + C²)
    numerator = (A * point.x + B * point.y + C * point.z + D)
    denominator = math.sqrt(A*A + B*B + C*C)
    
    if denominator < 1e-8:
        return 0.0  # 避免除以零
    
    return numerator / denominator

def count_inliers(points: List[Vector3], plane_coeffs: tuple, threshold: float) -> tuple:
    """
    统计在给定距离阈值内的内点数量和比例
    
    参数:
        points: 点集
        plane_coeffs: (A, B, C, D)平面方程系数
        threshold: 距离阈值
    
    返回:
        (inliers_count, inliers_ratio): 内点数量和比例
    """
    inliers_count = 0
    for point in points:
        distance = calculate_distance_to_plane(point, plane_coeffs)
        distance = abs(distance)
        if distance <= threshold:
            inliers_count += 1
    
    inliers_ratio = inliers_count / (float)(len(points)) if points else 0.0
    # print("inliers_count:", inliers_count, "inliers_ratio:", (float)(len(points)))
    
    return inliers_count, inliers_ratio

def ransac_plane_fitting(points: List[Vector3], iterations: int = 100, threshold: float = 0.1) -> tuple:
    """
    使用RANSAC算法拟合最优平面
    
    参数:
        points: 点集
        iterations: RANSAC迭代次数
        threshold: 距离阈值，用于判断内点
    
    返回:
        (best_plane, best_centroid, best_inliers_count, best_inliers_ratio): 
        最优平面系数、最优平面重心、内点数量和比例
    """
    if len(points) < 3:
        raise ValueError("点集数量至少为3")
    
    best_inliers_count = 0
    best_plane = None
    best_centroid = None
    best_inliers_ratio = 0.0
    
    # 执行RANSAC迭代
    for i in range(iterations):
        try:
            # 随机选择三个不同的点
            sample_indices = random.sample(range(len(points)), 3)
            p1, p2, p3 = points[sample_indices[0]], points[sample_indices[1]], points[sample_indices[2]]
            
            # 计算通过这三个点的平面
            A, B, C, D, centroid = calculate_plane_from_three_points(p1, p2, p3)
            plane_coeffs = (A, B, C, D)
            
            # 计算内点数量和比例
            inliers_count, inliers_ratio = count_inliers(points, plane_coeffs, threshold)
            
            # 更新最优平面
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_inliers_ratio = inliers_ratio
                best_plane = plane_coeffs
                best_centroid = centroid
                
            # 如果已经找到非常好的平面，可以提前退出
            if inliers_ratio >= 0.95:
                break
                
        except ValueError:
            # 处理三点共线的情况，跳过当前迭代
            continue
    
    # 如果没有找到有效平面，使用所有点拟合一个平面
    if best_plane is None and len(points) >= 3:
        A, B, C, D, centroid = fit_plane(points)
        best_plane = (A, B, C, D)
        best_centroid = centroid
        best_inliers_count, best_inliers_ratio = count_inliers(points, best_plane, threshold)
    print("best_inliers_count:", best_inliers_count, "best_inliers_ratio:", best_inliers_ratio)
    return best_plane, best_centroid, best_inliers_count, best_inliers_ratio

def determine_trajectory_cut_range(points: List[Vector3], best_plane: tuple, threshold: float) -> tuple:
    """
    根据最优平面确定轨迹的头尾切割范围
    
    参数:
        points: 原始点集
        best_plane: 最优平面系数 (A, B, C, D)
        threshold: 距离阈值
    
    返回:
        (start_idx, end_idx): 切割后的起始和结束索引
    """
    n = len(points)
    
    # 计算每个点到平面的距离
    distances = [calculate_distance_to_plane(point, best_plane) for point in points]
    
    # 找到连续内点区域的结束索引
    distance_sign = 1
    if (distances[-1] < 0):
        distance_sign = -1
    else:
        distance_sign = 1
    abs_distances = [abs(distance) for distance in distances]
    test_threshold = min(threshold, 0.25 * max(abs_distances))
    # 标记内点
    is_inlier = [distance <= test_threshold for distance in abs_distances]
    is_inlier_end = []
    if (distance_sign == 1):
        is_inlier_end = is_inlier
    else:
        is_inlier_end = [distance >= (-1) * test_threshold for distance in distances]
    # 找到连续内点区域的起始索引

    start_idx = max((int)(n*0.2), 1)
    while start_idx > 0  and is_inlier[start_idx]:
        start_idx -= 1

    end_idx = n-1
    while end_idx > (int)(n*0.8) and not is_inlier_end[end_idx]:
        end_idx -= 1
    # print("distance_sign:",distance_sign, "test_threshold", test_threshold, "start_idx:" , start_idx,"end_idx:" , end_idx, ",n:" , n)
    # 确保索引有效
    if start_idx > end_idx:
        # 如果没有足够的内点，返回整个范围
        start_idx = 0
        end_idx = n - 1
    
    return start_idx, end_idx

def ransac_trajectory_cutting(points: List[Vector3], ransac_iterations: int = 100, distance_threshold: float = 0.03) -> tuple:
    """
    使用RANSAC思想进行轨迹头尾切割
    
    参数:
        points: 原始轨迹点集
        ransac_iterations: RANSAC迭代次数
        distance_threshold: 距离阈值，用于判断内点
    
    返回:
        (filtered_points, best_normal, best_centroid): 
        切割后的轨迹点集、最优平面法向量和重心
    """
    n = len(points)
    
    if n < 10:  # 至少需要10个点才能去掉头尾10%
        raise ValueError("轨迹点数量至少为10")
    
    # 去掉头尾10%的点
    cut_percent = 0.2
    cut_count = int(n * cut_percent)
    middle_points = points[cut_count:-cut_count] if cut_count > 0 else points
    
    # 计算剩余点的重心
    middle_centroid = calculate_mean(middle_points)
    
    # 使用RANSAC拟合最优平面
    best_plane, best_centroid, best_inliers_count, best_inliers_ratio = \
        ransac_plane_fitting(middle_points, ransac_iterations, distance_threshold)
    
    # 如果RANSAC失败，使用中间点集的拟合结果
    if best_plane is None:
        best_plane, _, _, _, best_centroid = fit_plane(middle_points)
    
    # 计算法向量
    A, B, C, D = best_plane
    best_normal = Vector3(A, B, C)
    
    # 在原始点集中确定切割范围
    start_idx, end_idx = determine_trajectory_cut_range(points, best_plane, distance_threshold)
    
    # 获取切割后的轨迹点
    filtered_points = points[start_idx:end_idx+1] if start_idx <= end_idx else points
    
    best_plane, _, _, _, best_centroid = fit_plane(filtered_points)
    best_normal = Vector3(A, B, C)

    return filtered_points, best_normal

import random
def matrix_vector_mult(M, v):
    """矩阵乘向量"""
    return [M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]]

def solve_3x3(A, b):
    """
    高斯消元法求解3x3线性系统 Ax = b
    :param A: 3x3矩阵（列表的列表）
    :param b: 3维向量（列表）
    :return: 解向量x（列表）
    """
    # 构造增广矩阵
    M = [A[0] + [b[0]], A[1] + [b[1]], A[2] + [b[2]]]
    
    # 前向消元
    for i in range(3):
        # 部分枢轴选择
        max_row = max(range(i, 3), key=lambda r: abs(M[r][i]))
        M[i], M[max_row] = M[max_row], M[i]
        
        pivot = M[i][i]
        if abs(pivot) < 1e-10:
            raise ValueError("矩阵奇异，无法求解")
        
        for j in range(i+1, 3):
            factor = M[j][i] / pivot
            for k in range(i, 4):
                M[j][k] -= factor * M[i][k]
    
    # 回代
    x = [0] * 3
    for i in range(2, -1, -1):
        x[i] = M[i][3]
        for j in range(i+1, 3):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x

def fit_plane(points):
    """
    拟合最佳平面（最小二乘意义下点到平面距离和最小）
    :param points: List[Vector3]，三维点列表
    :return: (A, B, C, D) 平面方程系数（Ax + By + Cz + D = 0）
    """
    n = len(points)
    if n < 3:
        raise ValueError("至少需要3个点来拟合平面")
    
    # 1. 计算质心
    centroid = calculate_mean(points)
    
    # 2. 计算协方差矩阵（3x3）
    C = [[0.0] * 3 for _ in range(3)]
    for p in points:
        diff = p - centroid  # Vector3对象
        # 计算协方差矩阵元素（对称矩阵）
        C[0][0] += diff.x * diff.x  # Cxx
        C[0][1] += diff.x * diff.y  # Cxy
        C[0][2] += diff.x * diff.z  # Cxz
        C[1][0] += diff.y * diff.x  # Cyx (对称)
        C[1][1] += diff.y * diff.y  # Cyy
        C[1][2] += diff.y * diff.z  # Cyz
        C[2][0] += diff.z * diff.x  # Czx (对称)
        C[2][1] += diff.z * diff.y  # Czy (对称)
        C[2][2] += diff.z * diff.z  # Czz
    
    # 3. 逆迭代法求最小特征值对应的特征向量（平面法向量）
    # 初始化随机向量
    v = [random.random(), random.random(), random.random()]
    tolerance = 1e-8
    max_iter = 50
    
    for _ in range(max_iter):
        try:
            # 解线性系统 C w = v（等价于计算 C^{-1} v）
            w = solve_3x3(C, v)
        except ValueError:
            # 矩阵奇异时使用当前向量
            w = v
        
        # 归一化向量
        norm_w = math.sqrt(sum(x**2 for x in w))
        if norm_w < tolerance:
            break
        w_normalized = [x / norm_w for x in w]
        
        # 检查收敛（向量变化小于容差）
        diff_norm = math.sqrt(sum((w_normalized[i] - v[i])**2 for i in range(3)))
        if diff_norm < tolerance:
            v = w_normalized
            break
        
        v = w_normalized
    
    # v即为法向量 (A, B, C)
    A, B, C = v
    
    # 4. 计算平面常数项 D = - (A*x0 + B*y0 + C*z0)
    D = - (A * centroid.x + B * centroid.y + C * centroid.z)
    
    return A, B, C, D, centroid

def get_reverse_distance(points: Vector3) -> List[float]:
    center_point = calculate_mean(points)
    distance_to_plane = []
    for point in points:
        to_center = point - center_point
        distance_to_plane.append(abs(to_center.magnitude()))
    return distance_to_plane[::-1]
def detect_inflection_index(points,window_size = 5,threshold = 0.2):
    if len(points) < window_size:
        return 0
    inflection_index = 0

    start_dist = points[0]
    dist_w = points[window_size-1]
    start_slope = (start_dist - dist_w)/(window_size-1.0)
    # print(start_slope)
    end_slope = start_slope * threshold       
    for i in range(window_size, len(points) - window_size):
        
        window = points[i-window_size:i]
        # print(window)
        left_slope = (window[0] - window[-1]) / (window_size-1.0)
        if (left_slope < end_slope):
            inflection_index = i - window_size
            return inflection_index
    return inflection_index

def detect_speed_local_minimal(points:[Vector3]):
    speeds = []
    speeds_idx = []
    start_speed = (points[1] - points[0]).magnitude()
    if (start_speed > 0):
        speeds.append(start_speed)
        speeds_idx.append(0)
    for i in range(2,len(points)):
        if ((points[i] - points[i-1]).magnitude() == 0):
            continue
        speeds.append((points[i] - points[i-1]).magnitude())
        speeds_idx.append(i-1)
        # if ((points[i] - points[i-1]).magnitude() < start_speed):
        #     start_speed = (points[i] - points[i-1]).magnitude()
        # else:
            # return i
    start_idx = 0
    for i in range(1,len(speeds)):
        if (speeds[i] > speeds[i-1]):
            start_idx = speeds_idx[i-1]
            break
    return (speeds, start_idx)        

def delete_outliers(points: List[Vector3], puncture_plane_norm : Vector3, puncture_plane_center: Vector3) -> (List[Vector3],Vector3):
    if not points:
        return []

    start_thres = 0.3
    end_thres = 0.7
    # 切掉前后共50%
    chunk_points = points[int(start_thres * len(points)):int(end_thres* len(points))]
    # 计算中心点
    A,B,C,D,centroid = fit_plane(chunk_points)
    print("A,B,C,D,centroid:", A,B,C,D,centroid)
    print("puncture_plane_center, delta:", puncture_plane_center, (puncture_plane_center - centroid).magnitude())

    norm_cal = Vector3(A,B,C).normalized()
    ref_center = centroid
    print("norm, puncture_plane_norm:", norm_cal, puncture_plane_norm)

    # norm = puncture_plane_norm.normalized()
    norm = norm_cal

    to_plane_distances = []
    # 计算点到平面的距离和在平面内的半径
    for point in points:
        to_center = point - ref_center
        to_plane_distances.append(abs(norm.dot(to_center)))

    chunk_to_plane_distances = to_plane_distances[int(start_thres * len(points)):int(end_thres * len(points))]

    pairs = sorted(enumerate(chunk_to_plane_distances), key=lambda x: x[1])
    sorted_plane_dist = [v for _, v in pairs]
    sorted_indices = [i for i, _ in pairs]
    # top_40_percent = int(0.4 * len(sorted_plane_dist))
    cut_distance = sorted_plane_dist[-1]

    start_idx = 0
    for i in reversed(range(0,int(start_thres * len(points)))):
        if to_plane_distances[i] > cut_distance:
            print("check distance:",to_plane_distances[i])
            start_idx = i
            break
        
    end_idx = len(points)
    for i in range(int(end_thres * len(points)),len(points)):
        if to_plane_distances[i] > cut_distance:
            end_idx = i
            break
    print("start_idx, end_idx:", start_idx, end_idx)

    chunk_points = points[start_idx:end_idx]
    # 计算中心点
    A,B,C,D,centroid = fit_plane(chunk_points)
    print("A,B,C,D,centroid:", A,B,C,D,centroid)
    print("puncture_plane_center, delta:", puncture_plane_center, (puncture_plane_center - centroid).magnitude())

    norm_cal = Vector3(A,B,C).normalized()
    ref_center = centroid
    print("norm, puncture_plane_norm:", norm_cal, puncture_plane_norm)

    # norm = puncture_plane_norm.normalized()
    norm = norm_cal

    to_plane_distances = []
    # 计算点到平面的距离和在平面内的半径
    for point in points:
        to_center = point - ref_center
        to_plane_distances.append(abs(norm.dot(to_center)))

    chunk_to_plane_distances = to_plane_distances[start_idx:end_idx]

    pairs = sorted(enumerate(chunk_to_plane_distances), key=lambda x: x[1])
    sorted_plane_dist = [v for _, v in pairs]
    sorted_indices = [i for i, _ in pairs]
    # top_40_percent = int(0.4 * len(sorted_plane_dist))
    cut_distance = sorted_plane_dist[-1]

    for i in reversed(range(0,start_idx)):
        if to_plane_distances[i] > cut_distance:
            print("check distance:",to_plane_distances[i])
            start_idx = i
            break
        
    for i in range(end_idx,len(points)):
        if to_plane_distances[i] > cut_distance:
            end_idx = i
            break
    print("start_idx, end_idx:", start_idx, end_idx)


    # start_idx = int(0.25 * len(points))
    # end_idx = int(0.75 * len(points))
    return (points[start_idx:end_idx],norm_cal)

    # # 计算法向量和参考中心
    # norm = puncture_plane_norm.normalized()
    # ref_center = puncture_plane_center

    # to_plane_distances = []
    # # 计算点到平面的距离和在平面内的半径
    # for point in points:
    #     to_center = point - ref_center
    #     to_plane_distances.append(abs(norm.dot(to_center)))

    # pairs = sorted(enumerate(to_plane_distances), key=lambda x: x[1])
    # sorted_plane_dist = [v for _, v in pairs]
    # sorted_indices = [i for i, _ in pairs]

    # top_20_percent = int(0.2 * len(sorted_plane_dist))
    # top_40_percent = int(0.4 * len(sorted_plane_dist))

    # nearpoints = []
    # for i in sorted_indices[:top_20_percent]:
    #     nearpoints.append(points[i])

    # nearCenter = calculate_mean(nearpoints)

    # to_plane_distances = []
    # for point in points:
    #     to_center = point - nearCenter
    #     to_plane_distances.append(abs(norm.dot(to_center)))

    # pairs = sorted(enumerate(to_plane_distances), key=lambda x: x[1])
    # sorted_plane_dist = [v for _, v in pairs]

    # # distance_minimal = float("inf")
    # # distance_minimal_idx = 0
    # # for i in range(top_20_percent):
    # #     if to_plane_distances[i] < distance_minimal:
    # #         distance_minimal = to_plane_distances[i]
    # #         distance_minimal_idx = i
    
    # # print("distance_minimal,distance_minimal_idx:", distance_minimal, distance_minimal_idx)

    # min_mean = sum(sorted_plane_dist[:top_20_percent]) / top_20_percent
    # cut_distance = max(
    #     abs(sorted_plane_dist[0] - min_mean),
    #     abs(sorted_plane_dist[top_20_percent-1] - min_mean)
    # )

    # print("size:", len(sorted_plane_dist))
    # print("min_mean, cut_distance:", min_mean, cut_distance)

    # start_idx = 0
    # for i in (range(len(points))):
    #     if to_plane_distances[i] < cut_distance:
    #         print("check distance:",to_plane_distances[i])
    #         start_idx = i
    #         break
        
    # end_idx = len(points)
    # for i in reversed(range(len(points))):
    #     if to_plane_distances[i] < cut_distance:
    #         end_idx = i
    #         break
    # print("start_idx, end_idx:", start_idx, end_idx)

    # return points[start_idx:end_idx]
    # # result = []
    # # for i in (range(len(points))):
    # #     if to_plane_distances[i] < cut_distance:
    # #         result.append(points[i])
    # # return result
    
## old check functin: untested
def clean_trajectory_check(points, puncture_plane_norm):
    if not points:
        return False

    # 计算平面参数
    all_center = calculate_mean(points)
    norm = puncture_plane_norm.normalized()

    buffer_count = len(points)
    tolerance_count = int(buffer_count * 0.2)
    vectors_in_plane = []

    # 投影到平面
    for point in points:
        to_center = point - all_center
        to_plane = (norm.dot(to_center) * norm)
        in_plane = to_center - to_plane
        vectors_in_plane.append(in_plane)

    # 平面内分析
    all_center_in_plane = calculate_mean(vectors_in_plane)
    norm_in_plane = norm  # 假设平面法向量与原始平面相同

    cross_directions = []
    moving_angle = 0
    prev_vector = Vector3(0,0,0)
    first_vector = Vector3(0,0,0)
    end_vector = Vector3(0,0,0)

    for i in range(buffer_count):
        to_center = points[i] - all_center_in_plane
        to_plane = (norm_in_plane.dot(to_center) * norm_in_plane)
        current_vector = to_center - to_plane

        if i == 0:
            prev_vector = current_vector
            first_vector = current_vector
            continue

        # 计算移动角度
        angle = math.degrees(math.acos(
            current_vector.normalized().dot(prev_vector.normalized())
        ))
        moving_angle += angle

        # 计算叉积方向
        cross = current_vector.cross(prev_vector)
        cross_dir = 1 if cross.dot(norm_in_plane) > 0 else -1
        cross_directions.append(cross_dir)

        # 检测下降幅度
        if current_vector.magnitude() < prev_vector.magnitude() * 0.9:
            tolerance_count -= 1
            if tolerance_count < 0:
                print("小幅下降过多错误")
                return False

        prev_vector = current_vector
        end_vector = current_vector

    # 转向分析
    move_length = 0
    plus_move = 0
    minus_move = 0
    turn_thres = 15

    for i in range(1, len(cross_directions)):
        move_length += 1
        if cross_directions[i] * cross_directions[i-1] < 0:
            if move_length > turn_thres:
                if cross_directions[i-1] > 0: plus_move += 1
                else: minus_move += 1
            move_length = 0

    # 最终检查
    if move_length > turn_thres:
        if cross_directions[-1] > 0: plus_move += 1
        else: minus_move += 1

    # 判断转向一致性
    if (plus_move >= minus_move and minus_move > 0) or (minus_move > plus_move and plus_move > 0):
        print(f"转向不一致错误: plus_move={plus_move}, minus_move={minus_move}")
        return False

    # 整体增长检查
    overall_growth = (end_vector.magnitude() - first_vector.magnitude()) / first_vector.magnitude()
    if overall_growth < -0.15:
        print(f"从内到外错误: {overall_growth}")
        return False

    return True



