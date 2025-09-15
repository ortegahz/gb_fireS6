import numpy as np
from scipy.spatial.distance import cdist


def find_outliers(points, threshold=3.0, max_outlier_num=3):
    """
    找出二维点集中最离群的多个点（基于到质心的距离）

    参数:
    points: 二维点列表，格式为[(x1,y1), (x2,y2), ...]
    threshold: 阈值，如果最远距离小于阈值，则不认为是真离群点
    max_outlier_num: 最多删除的离群点数量

    返回:
    outlier_indices: 离群点的索引列表
    is_true_outliers: 是否为真离群点的布尔列表
    max_distances: 最大距离值列表
    """
    if len(points) == 0:
        return [], [], []

    outliers_indices = []
    is_true_outliers = []
    max_distances = []

    # 创建点的副本，避免修改原始数据
    remaining_points = points.copy()
    remaining_indices = list(range(len(points)))

    for _ in range(max_outlier_num):
        if len(remaining_points) <= 1:  # 如果只剩一个点或没有点，停止删除
            break

        # 计算质心（所有剩余点的平均值）
        centroid = np.mean(remaining_points, axis=0)

        # 计算每个点到质心的欧氏距离
        distances = cdist([centroid], remaining_points, 'euclidean')[0]

        # 找到距离最大的点（最离群点）
        outlier_idx_in_remaining = np.argmax(distances)
        max_distance = distances[outlier_idx_in_remaining]

        # 判断是否为真离群点
        is_true_outlier = max_distance > threshold

        if is_true_outlier:
            # 获取原始索引
            original_index = remaining_indices[outlier_idx_in_remaining]

            # 添加到结果列表
            outliers_indices.append(original_index)
            is_true_outliers.append(True)
            max_distances.append(max_distance)

            # 从剩余点中删除该离群点
            del remaining_points[outlier_idx_in_remaining]
            del remaining_indices[outlier_idx_in_remaining]
        else:
            # 如果没有找到真离群点，停止循环
            break

    return outliers_indices, is_true_outliers, max_distances


def outlier_filter(res, min_valid_num=(3, 2), threshold=2.5,  max_outlier_num=3, verbose=False):
    ret_flag = True
    # 去除权重为0的无效坐标
    res_valid = [r for r in res if r[0] != 0]
    if len(res_valid) == 0:
        ret_flag = False
        return ret_flag, (0.0, 0.0), 0, 0
    points = list(np.array(res_valid)[:, 1])
    if verbose:
        print("所有有效点:")
        for i, (x, y) in enumerate(points):
            print(f"点 {i}: ({x:.2f}, {y:.2f})")
    non_zero_num = len(points)
    if len(points) < min_valid_num[0]:
        if verbose:
            print(f"\n删除无效点后剩余 {len(points)} 个点，不足{min_valid_num[0]}，不予报警")
        ret_flag = False
        # return ret_flag, (0.0, 0.0)

    # 查找离群点
    outlier_indices, is_true_list, max_dists = find_outliers(points, threshold, max_outlier_num)

    if verbose:
        print(f"\n找到 {len(outlier_indices)} 个真离群点:")
        for i, idx in enumerate(outlier_indices):
            print(f"离群点索引 {idx}: 坐标({points[idx][0]:.2f}, {points[idx][1]:.2f}), "
                  f"距离 {max_dists[i]:.2f}, 是否真离群: {is_true_list[i]}")

    # 删除真离群点
    if outlier_indices:
        # 按索引从大到小排序，避免删除时索引变化
        sorted_indices = sorted(outlier_indices, reverse=True)
        cleaned_res = res_valid.copy()

        for idx in sorted_indices:
            del cleaned_res[idx]

        if verbose:
            print(f"\n删除离群点后剩余 {len(cleaned_res)} 个点:")
            for i, (conf, coord) in enumerate(cleaned_res):
                print(f"点 {i}: 权重{conf}, 坐标({coord[0]}, {coord[1]})")
    else:
        cleaned_res = res_valid

        if verbose:
            print("\n没有找到需要删除的真离群点")
    non_outlier_num =len(cleaned_res)

    if len(cleaned_res) < min_valid_num[1]:

        if verbose:
            print(f"\n删除离群点后剩余 {len(cleaned_res)} 个点，不足{min_valid_num[1]}，不予报警")
        ret_flag = False
        # return ret_flag, (0.0, 0.0)

    # 计算加权平均坐标
    final_coord = np.array([0.0, 0.0])
    total_conf = 0
    for cp in cleaned_res:
        conf, coord = cp
        total_conf += conf
        final_coord += conf * np.array(coord)
    if total_conf == 0:
        ret_flag = False
        return ret_flag, (0.0, 0.0), non_zero_num, non_outlier_num

    weighted_avg = final_coord / total_conf
    if verbose:
        print('\n删除无效点、删除离群点后，加权平均的坐标为：')
        print(f"({weighted_avg[0]:.2f}, {weighted_avg[1]:.2f})")
    return ret_flag, weighted_avg, non_zero_num, non_outlier_num

# 示例使用
if __name__ == "__main__":
    # 权重和坐标
    res = [
        (0.9, [25, 10]),
        (0.4, [25, 6]),
        (0.7, [26, 12]),
        (0.3, [25, 9]),
        (0.0, [0, 0]),
        (0.6, [26, 10]),
        (0.5, [25, 14]),
        (0.8, [28, 10]),
    ]
    outlier_filter(res, verbose=True)