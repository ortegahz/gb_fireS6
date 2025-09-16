#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两个 output.txt 文件的一致性。

output.txt 文件格式:
frame_id\tfilename\t[Fire(...), Fire(...)]

本脚本执行以下对比:
1.  以 frame_id 为主键对齐两个文件。
2.  统计各自独有的帧（即只在一个文件中出现告警）。
3.  对于共有帧，对比告警数量是否一致。
4.  若数量一致，通过 IoU 对告警框进行配对，并对比 score 和 alarm_flag。
5.  计算 score 的误差指标 (RMSE, MAE, Corr)。
6.  对完美匹配的告警，计算 center_point 的平均误差、方差和最大误差。
7.  生成可视化图表，并输出差异明细。
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_iou(box1, box2):
    """计算两个边界框(x, y, w, h)的IoU"""
    # 转换成 (x1, y1, x2, y2) 格式
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # 计算交集区域坐标
    xi_1, yi_1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi_2, yi_2 = min(x2_1, x2_2), min(y2_1, y2_2)

    # 计算交集面积
    inter_width = max(0, xi_2 - xi_1)
    inter_height = max(0, yi_2 - yi_1)
    inter_area = inter_width * inter_height

    # 计算并集面积
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def parse_fire_objects(fire_str):
    """从字符串中解析出Fire对象列表，支持嵌套括号和不同格式。"""
    if not fire_str or fire_str.strip() == '[]':
        return []

    objects = []
    # Regex patterns for the fields we need to compare.
    # Making them robust to scientific notation (e/E).
    box_pattern = re.compile(r"fire_box=\(([\d\s.,eE+-]+)\)")
    score_pattern = re.compile(r"score=([\d.eE+-]+)")
    flag_pattern = re.compile(r"alarm_flag=(True|False)")
    # 修改: 兼容 array([...]) 和 (...) 两种格式
    point_pattern_array = re.compile(r"center_point=array\(\[([\d\s.,eE+-]+)\]\)")
    point_pattern_tuple = re.compile(r"center_point=\(([\d\s.,eE+-]+)\)")

    # Manually find each "Fire(...)" block to handle nested parentheses correctly.
    # This is more robust than a single complex regex.
    search_start = 0
    while True:
        try:
            # Find the start of the content, right after "Fire("
            content_start_idx = fire_str.index("Fire(", search_start) + len("Fire(")

            # Scan to find the matching closing parenthesis for this "Fire(...)" block
            paren_level = 1
            scan_idx = content_start_idx
            while scan_idx < len(fire_str) and paren_level > 0:
                char = fire_str[scan_idx]
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                scan_idx += 1

            # If we didn't find a matching parenthesis, the string is malformed.
            if paren_level != 0:
                break

            # Extract the content of this Fire object
            content_end_idx = scan_idx - 1
            obj_str = fire_str[content_start_idx:content_end_idx]

            # Set the next search to start after this object.
            search_start = scan_idx

            # Now, parse the fields from the extracted object content string
            box_match = box_pattern.search(obj_str)
            score_match = score_pattern.search(obj_str)
            flag_match = flag_pattern.search(obj_str)

            if not all([box_match, score_match, flag_match]):
                continue

            try:
                # Parse fire_box: "x, y, w, h" -> tuple of floats
                box_coords_str = box_match.group(1).split(',')
                fire_box = tuple(map(float, [s.strip() for s in box_coords_str]))
                score = float(score_match.group(1))
                alarm_flag = flag_match.group(1) == 'True'

                # 修改: 兼容两种格式解析 center_point
                center_point = (0.0, 0.0)
                point_match = point_pattern_array.search(obj_str)
                if not point_match:
                    point_match = point_pattern_tuple.search(obj_str)

                if point_match:
                    point_coords_str = point_match.group(1).split(',')
                    center_point = tuple(map(float, [s.strip() for s in point_coords_str]))
                else:
                    # 如果两种格式都匹配不到，才打印警告
                    print(f"Warning: center_point not found in object string: {obj_str[:150]}")

                objects.append({
                    "fire_box": fire_box,
                    "score": score,
                    "alarm_flag": alarm_flag,
                    "center_point": center_point,
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse values from object string chunk: {obj_str[:150]}. Error: {e}")
                continue
        except ValueError:
            # No more "Fire(" substrings found. We are done.
            break

    return objects


def load_txt(path):
    """
    读取 output.txt 并解析内容。
    返回 dict: key=frame_id -> value=[{fire_box, score, alarm_flag}, ...]
    """
    res = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            try:
                frame_id = int(parts[0])
                fire_objects = parse_fire_objects(parts[2])
                # 只存储有告警的帧
                if fire_objects:
                    res[frame_id] = fire_objects
            except ValueError:
                print(f"Warning: Could not parse frame_id from line: {line.strip()}")
                continue
    return res


def compute_score_metrics(scores1, scores2):
    """计算 score 的误差指标"""
    if not scores1 or not scores2:
        return dict(MSE=np.nan, RMSE=np.nan, MAE=np.nan, Corr=np.nan)

    v1 = np.array(scores1)
    v2 = np.array(scores2)

    mse = mean_squared_error(v1, v2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(v1, v2)
    corr = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else 1.0 if len(v1) == 1 else np.nan
    return dict(MSE=mse, RMSE=rmse, MAE=mae, Corr=corr)


def main():
    parser = argparse.ArgumentParser(description="Compare two output.txt files for fire detection results.")
    parser.add_argument("--file_a", default="/home/manu/tmp/output_gb_s6_py_v0.txt",
                        help="Path to the first output file (e.g., ground truth or baseline).")
    parser.add_argument("--file_b", default="/home/manu/tmp/output_gb_s6_cpp.txt",
                        help="Path to the second output file (to be compared).")
    parser.add_argument("--iou_thresh", type=float, default=0.7, help="IoU threshold for matching boxes.")
    parser_add_boolean = lambda p, name, default, help: p.add_argument(f'--{name}', f'--no-{name}', help=help,
                                                                       default=default,
                                                                       action=argparse.BooleanOptionalAction)
    parser_add_boolean(parser, "show", default=True, help="Show the plot after generating.")
    parser.add_argument("--out_png", default="/home/manu/tmp/compare_fire_detection.png",
                        help="Path to save the comparison plot.")
    args = parser.parse_args()

    if not os.path.isfile(args.file_a):
        raise FileNotFoundError(f"File A not found: {args.file_a}")
    if not os.path.isfile(args.file_b):
        raise FileNotFoundError(f"File B not found: {args.file_b}")

    print("Loading and parsing files...")
    res_a = load_txt(args.file_a)
    res_b = load_txt(args.file_b)

    keys_a = set(res_a.keys())
    keys_b = set(res_b.keys())
    common_keys = sorted(list(keys_a & keys_b))
    only_a = sorted(list(keys_a - keys_b))
    only_b = sorted(list(keys_b - keys_a))

    print(f"\n--- Overview ---")
    print(f"Frames with alarms in both files (Common): {len(common_keys)}")
    print(f"Frames with alarms only in A: {len(only_a)}")
    print(f"Frames with alarms only in B: {len(only_b)}")

    # --------- Core Comparison Logic ---------
    perfect_matches = []
    count_mismatches = []
    content_mismatches = []

    paired_scores_a = []
    paired_scores_b = []
    # 新增: 用于存储配对成功的 center_point
    paired_points_a = []
    paired_points_b = []
    # 新增: 用于存储详细的点位误差信息以供打印
    point_error_details = []

    for frame_id in common_keys:
        alarms_a = res_a[frame_id]
        alarms_b = res_b[frame_id]

        # 1. Check alarm count
        if len(alarms_a) != len(alarms_b):
            count_mismatches.append(frame_id)
            continue

        # 2. Match alarms using IoU and compare content
        is_perfect_frame = True
        matched_b_indices = set()
        frame_scores_a = []
        frame_scores_b = []
        # 新增: 存储当前帧的配对点
        frame_points_a = []
        frame_points_b = []

        # Create a cost matrix (1 - IoU)
        num_alarms = len(alarms_a)
        iou_matrix = np.zeros((num_alarms, num_alarms))
        for i in range(num_alarms):
            for j in range(num_alarms):
                iou_matrix[i, j] = calculate_iou(alarms_a[i]["fire_box"], alarms_b[j]["fire_box"])

        # Greedy matching based on highest IoU
        a_indices = list(range(num_alarms))
        b_indices = list(range(num_alarms))

        pairs = []
        # Find best match for each alarm in A
        for i in a_indices:
            best_j = -1
            max_iou = -1
            for j in b_indices:
                if j in matched_b_indices:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    best_j = j
            if max_iou > args.iou_thresh:
                pairs.append((i, best_j))
                matched_b_indices.add(best_j)

        if len(pairs) != num_alarms:
            is_perfect_frame = False
            content_mismatches.append({
                "frame_id": frame_id,
                "reason": f"Box matching failed. Found {len(pairs)} pairs with IoU>{args.iou_thresh}, expected {num_alarms}."})
            continue

        # Now check content for matched pairs
        for i, j in pairs:
            alarm_a = alarms_a[i]
            alarm_b = alarms_b[j]

            # Compare score (with a small tolerance) and alarm_flag
            score_diff = abs(alarm_a["score"] - alarm_b["score"])
            flag_match = alarm_a["alarm_flag"] == alarm_b["alarm_flag"]

            if score_diff > 1e-6 or not flag_match:
                is_perfect_frame = False
                content_mismatches.append({
                    "frame_id": frame_id,
                    "reason": f"Content diff on paired boxes.",
                    "box_a": alarm_a['fire_box'], "score_a": alarm_a['score'], "flag_a": alarm_a['alarm_flag'],
                    "box_b": alarm_b['fire_box'], "score_b": alarm_b['score'], "flag_b": alarm_b['alarm_flag'],
                    "iou": iou_matrix[i, j]
                })
                break  # A single content mismatch invalidates the frame
            else:
                # This pair is a perfect match
                frame_scores_a.append(alarm_a["score"])
                frame_scores_b.append(alarm_b["score"])
                # 新增: 收集center_point用于后续分析
                frame_points_a.append(alarm_a['center_point'])
                frame_points_b.append(alarm_b['center_point'])

        if is_perfect_frame:
            perfect_matches.append(frame_id)
            paired_scores_a.extend(frame_scores_a)
            paired_scores_b.extend(frame_scores_b)
            # 新增: 如果整帧都完美匹配，则将点位信息加入总列表
            paired_points_a.extend(frame_points_a)
            paired_points_b.extend(frame_points_b)
            # 新增: 记录详细的点位误差信息
            for idx in range(len(frame_points_a)):
                pa = frame_points_a[idx]
                pb = frame_points_b[idx]
                dist = np.linalg.norm(np.array(pa) - np.array(pb))
                point_error_details.append({
                    "frame_id": frame_id,
                    "point_a": pa,
                    "point_b": pb,
                    "distance": dist
                })

    print("\n--- Common Frames Analysis ---")
    print(f"Perfect Matches: {len(perfect_matches)}")
    print(f"Count Mismatches: {len(count_mismatches)}")
    print(f"Content Mismatches: {len(content_mismatches)}")

    score_metrics = compute_score_metrics(paired_scores_a, paired_scores_b)
    print("\nScore Error Metrics (on perfectly matched alarms):")
    for k, v in score_metrics.items():
        print(f"  {k}: {v:.6f}")

    # 新增: 打印详细的点位误差，并计算最终统计值
    if point_error_details:
        print("\n--- Detailed Center Point Errors (Top 10 largest errors on perfectly matched alarms) ---")
        # 按误差从大到小排序
        sorted_errors = sorted(point_error_details, key=lambda x: x['distance'], reverse=True)
        for item in sorted_errors[:10]:
            pa_str = f"({item['point_a'][0]:.2f}, {item['point_a'][1]:.2f})"
            pb_str = f"({item['point_b'][0]:.2f}, {item['point_b'][1]:.2f})"
            print(f"  Frame {item['frame_id']}: A={pa_str:<18} | B={pb_str:<18} | Distance={item['distance']:.4f}")

    if paired_points_a:
        points_a = np.array(paired_points_a)
        points_b = np.array(paired_points_b)
        # 计算每对点之间的欧氏距离
        distances = np.linalg.norm(points_a - points_b, axis=1)
        mean_error = np.mean(distances)
        variance_error = np.var(distances)
        max_error = np.max(distances) if distances.size > 0 else 0

        # 新增: 计算超过1像素误差的数量和百分比
        errors_over_1_pixel = (distances > 1).sum()
        total_pairs = len(distances)
        percentage_over_1 = (errors_over_1_pixel / total_pairs) * 100 if total_pairs > 0 else 0

        print("\nCenter Point Error Metrics (on perfectly matched alarms):")
        print(f"  Mean Euclidean Distance: {mean_error:.6f}")
        print(f"  Variance of Euclidean Distance: {variance_error:.6f}")
        print(f"  Max Euclidean Distance: {max_error:.6f}")
        print(f"  Errors > 1.0 pixel: {errors_over_1_pixel} / {total_pairs} ({percentage_over_1:.2f}%)")

    # --------- Visualization ---------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Comparison: {os.path.basename(args.file_a)} vs {os.path.basename(args.file_b)}', fontsize=16)

    # Plot 1: Frame-level match status
    match_status = {}
    all_common_frames = sorted(keys_a | keys_b)
    status_values = []

    # Define status codes
    STATUS_PERFECT = 3
    STATUS_CONTENT_MISMATCH = 2
    STATUS_COUNT_MISMATCH = 1
    STATUS_ONLY_A = 0
    STATUS_ONLY_B = -1

    for frame in all_common_frames:
        if frame in perfect_matches:
            status_values.append(STATUS_PERFECT)
        elif any(d['frame_id'] == frame for d in content_mismatches):
            status_values.append(STATUS_CONTENT_MISMATCH)
        elif frame in count_mismatches:
            status_values.append(STATUS_COUNT_MISMATCH)
        elif frame in only_a:
            status_values.append(STATUS_ONLY_A)
        elif frame in only_b:
            status_values.append(STATUS_ONLY_B)

    ax1.step(all_common_frames, status_values, where='mid', label='Frame Match Status')
    ax1.set_yticks(
        ticks=[-1, 0, 1, 2, 3],
        labels=['Only in B', 'Only in A', 'Count Mismatch', 'Content Mismatch', 'Perfect Match']
    )
    ax1.set_ylabel("Match Status")
    ax1.set_title("Frame-level Match Status")
    ax1.grid(alpha=0.4, axis='y')
    ax1.set_xlabel("Frame ID")

    # Plot 2: Score difference for paired alarms
    if paired_scores_a:
        score_diff = np.array(paired_scores_a) - np.array(paired_scores_b)
        ax2.plot(score_diff, 'o-', markersize=3, alpha=0.7, label='Score Difference (A - B)')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel("Score Difference")
        ax2.set_xlabel("Paired Alarm Index")
        ax2.set_title(f"Score Difference for {len(score_diff)} Perfectly Paired Alarms")
        ax2.grid(alpha=0.3)
        ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out_png, dpi=150)
    print(f"\nComparison plot saved to: {args.out_png}")
    if args.show:
        plt.show()

    # --------- Print Detailed Differences ---------
    print("\n--- Detailed Differences (showing up to 5) ---")
    if only_a:
        print(f"\n*** Frames with alarms ONLY in A ({len(only_a)} total):")
        print(only_a[:5])
    if only_b:
        print(f"\n*** Frames with alarms ONLY in B ({len(only_b)} total):")
        print(only_b[:5])
    if count_mismatches:
        print(f"\n*** Frames with ALARM COUNT mismatch ({len(count_mismatches)} total):")
        for frame_id in count_mismatches[:5]:
            print(f"  Frame {frame_id}: A has {len(res_a[frame_id])} alarms, B has {len(res_b[frame_id])} alarms")
    if content_mismatches:
        print(f"\n*** Frames with CONTENT mismatch ({len(content_mismatches)} total):")
        for item in content_mismatches[:5]:
            print(f"  Frame {item['frame_id']}: {item['reason']}")
            if 'score_a' in item:
                print(
                    f"    - A: score={item['score_a']:.2f}, flag={item['flag_a']} | B: score={item['score_b']:.2f}, flag={item['flag_b']} (IoU: {item['iou']:.3f})")

    print("\nComparison finished.")


if __name__ == "__main__":
    main()
