import re
import json

def extract_vector3_from_log(log_line):
    """
    从日志行中提取两组Vector3坐标数据
    
    Args:
        log_line (str): 包含Puncture信息的日志行
        
    Returns:
        list: 包含两个Vector3坐标的列表，每个Vector3是一个包含x, y, z的字典
    """
    # 使用正则表达式匹配Puncture部分
    puncture_match = re.search(r'Puncture:\(([^)]+)\),\(([^)]+)\)', log_line)
    
    if not puncture_match:
        return None
    
    # 提取两个坐标组
    vectors_str = puncture_match.groups()
    vectors = []
    
    for vec_str in vectors_str:
        # 分割坐标值并转换为浮点数
        coords = vec_str.split(',')
        if len(coords) == 3:
            try:
                x = float(coords[0].strip())
                y = float(coords[1].strip())
                z = float(coords[2].strip())
                vectors.append({'x': x, 'y': y, 'z': z})
            except ValueError:
                # 处理数值转换错误
                continue
    
    return vectors

# 测试示例
if __name__ == "__main__":
    # 示例日志行
    log_line = "[2025-10-31 10:50:10.593] [Log] [Hand] Puncture:(0.1894, 0.5878, -0.7865),(-0.3751, 0.7295, 1.0766)"
    
    # 提取Vector3数据
    vectors = extract_vector3_from_log(log_line)
    
    if vectors:
        print("提取到的Vector3数据:")
        for i, vec in enumerate(vectors):
            print(f"Vector3_{i+1}: x={vec['x']}, y={vec['y']}, z={vec['z']}")
        
        # 输出为JSON格式便于集成
        print("\nJSON格式:")
        print(json.dumps(vectors, indent=2))
    else:
        print("未能提取到有效数据")