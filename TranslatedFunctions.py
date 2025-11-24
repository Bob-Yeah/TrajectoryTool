from typing import List, Tuple
import math
import random
from HandPR import Vector3, calculate_mean, fit_plane, calculate_gaussian_parameters

# 模拟CalculateOperationPlane函数，返回中心点和平面参数
def calculate_operation_plane(points: List[Vector3]) -> Tuple[Vector3, Tuple[float, float, float, float]]:
    """计算操作平面，返回中心点和平面参数(A, B, C, D)"""
    if not points:
        raise ValueError("点列表不能为空")
    
    # 使用HandPR.py中的calculate_mean获取中心点
    center = calculate_mean(points)
    
    # 使用HandPR.py中的fit_plane获取平面参数
    A, B, C, D, _ = fit_plane(points)
    
    return center, (A, B, C, D)

# 模拟DataRecorder功能的简单日志记录器
class Logger:
    @staticmethod
    def log(data_type: str, message: str):
        print(f"[{data_type}] {message}")

# 结构体SmallCircle的Python实现
class SmallCircle:
    def __init__(self, center: Vector3, radius: float):
        self.center = center
        self.radius = radius
    
    def contains_point(self, point: Vector3) -> bool:
        """检查点是否在圆内"""
        return (point - self.center).magnitude() < self.radius

# GetPerpendicular函数的Python实现
def get_perpendicular(vec: Vector3) -> Vector3:
    """获取垂直于给定向量的向量"""
    # 定义参考向量
    reference = Vector3(1, 0, 0)  # Vector3.right
    
    # 计算叉积的平方长度，如果太小则选择另一个参考向量
    cross = vec.cross(reference)
    if cross.magnitude() < 1e-6:
        reference = Vector3(0, 0, 1)  # Vector3.forward
    
    # 返回归一化的叉积
    return vec.cross(reference).normalized()

# RotateAroundAxis函数的Python实现
def rotate_around_axis(vector: Vector3, axis: Vector3, angle_rad: float) -> Vector3:
    """绕指定轴旋转向量"""
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    cross_product = axis.cross(vector)
    
    # Rodrigues' rotation formula
    rotated = Vector3(
        vector.x * cos_theta + cross_product.x * sin_theta,
        vector.y * cos_theta + cross_product.y * sin_theta,
        vector.z * cos_theta + cross_product.z * sin_theta
    )
    
    return rotated

# TrajectoryCoverage函数的Python实现
def trajectory_coverage(points: List[Vector3], norm: Vector3, center: Vector3, 
                       max_radius: float, radius: float = 0.02, samples: int = 1000) -> float:
    """计算轨迹覆盖率"""
    # 创建小圆列表
    circles = [SmallCircle(p, radius) for p in points]
    
    # 生成蒙特卡洛采样点
    in_count = 0
    vertical_vec = get_perpendicular(norm)
    
    for _ in range(samples):
        # 生成随机半径和角度
        sample_r = math.sqrt(random.random()) * max_radius
        sample_angle = random.random() * 2 * math.pi
        
        # 生成采样点
        sample_point = rotate_around_axis(vertical_vec.normalized(), norm.normalized(), sample_angle)
        sample_point = Vector3(
            center.x + sample_point.x * sample_r,
            center.y + sample_point.y * sample_r,
            center.z + sample_point.z * sample_r
        )
        
        # Logger.log("Hand", f"蒙特卡洛散点: {sample_point.x:.4f}, {sample_point.y:.4f}, {sample_point.z:.4f}")
        
        # 检查点是否在任何小圆内
        for circle in circles:
            if circle.contains_point(sample_point):
                in_count += 1
                break
    
    return in_count / samples

# CleanTrajectoryCheck2函数的Python实现
def clean_trajectory_check2(points: List[Vector3], puncture_plane_norm: Vector3) -> bool:
    """检查轨迹是否干净"""
    good_flag = True
    if not points:
        return False
    
    # 计算操作平面
    all_center, _ = calculate_operation_plane(points)
    norm = puncture_plane_norm.normalized()
    
    buffer_count = len(points)
    drop_count = 0  # 记录下降次数
    tolerance_count = int(buffer_count * 0.2)  # 调控参数
    
    vectors_in_plane = []
    for i in range(buffer_count):
        to_center = points[i] - all_center
        to_plane = Vector3(
            norm.x * norm.dot(to_center),
            norm.y * norm.dot(to_center),
            norm.z * norm.dot(to_center)
        )
        in_plane = to_center - to_plane
        
        # Logger.log("Hand", f"平面内点: {in_plane.x:.4f}, {in_plane.y:.4f}, {in_plane.z:.4f}")
        vectors_in_plane.append(in_plane)
    
    # 计算平面内点的中心点
    all_center_in_plane, _ = calculate_operation_plane(vectors_in_plane)
    
    first_vector_in_plane = None
    end_vector_in_plane = None
    prev_vector_in_plane = None
    radius = []
    first_radius = float('inf')
    
    buffer_count = len(vectors_in_plane)
    for i in range(buffer_count):
        to_center = vectors_in_plane[i] - all_center_in_plane
        radius.append(to_center.magnitude())
        
        if i == 0:
            first_radius = to_center.magnitude()
            prev_vector_in_plane = to_center
            first_vector_in_plane = to_center
            continue
        
        current = to_center.magnitude()
        prev = prev_vector_in_plane.magnitude()
        
        if current < prev * 0.9:  # 调控参数
            drop_count += 1
            if drop_count > tolerance_count:
                # 小幅下降次数过多
                good_flag = False
                Logger.log("Hand", f"小幅下降次数过多：{drop_count} 次，超过允许的 {tolerance_count} 次")
        
        prev_vector_in_plane = to_center
        end_vector_in_plane = to_center
    
    gaussian_center, std_radius = calculate_gaussian_parameters(radius)
    Logger.log("Hand", f"高斯中心: {gaussian_center:.4f}")
    Logger.log("Hand", f"标准差半径: {std_radius:.4f}")

    # 对半径排序
    radius_sorted = sorted(radius)
    
    percentile_20 = int(len(radius_sorted) * 0.3)
    
    Logger.log("Hand", f"从中心点开始？{first_radius}, 30分位半径: {radius_sorted[percentile_20]}")
    if first_radius > radius_sorted[percentile_20]:
        good_flag = False
        Logger.log("Hand", "消毒错误，没有从中心点开始")
    
    # 计算整体增长
    overall_growth = (end_vector_in_plane.magnitude() - first_vector_in_plane.magnitude()) / first_vector_in_plane.magnitude()
    
    if overall_growth < -0.15:  # 调控参数
        Logger.log("Hand", f"整体趋势不满足递增：从 {first_vector_in_plane.magnitude()} 到 {end_vector_in_plane.magnitude()}，变化 {overall_growth:.2%}")
        good_flag = False
    
    # 计算覆盖率
    coverage = trajectory_coverage(vectors_in_plane, norm, all_center_in_plane, radius_sorted[-1])
    
    if coverage < 0.7:  # 调控参数
        Logger.log("Hand", f"整体面积覆盖率：{coverage} 不满足：70%")
        good_flag = False
    
    Logger.log("Hand", f"趋势分析：整体增长 {overall_growth:.2%}，小幅下降 {drop_count} 次,占{(drop_count/buffer_count):.2%}，面积覆盖: {coverage}")
    
    if good_flag:
        Logger.log("Hand", "最终判定成功")
    else:
        Logger.log("Hand", "最终判定失败,具体原因见上部")
    
    return good_flag, end_vector_in_plane.magnitude(), all_center_in_plane