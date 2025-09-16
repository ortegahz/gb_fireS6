# from fire_loc_spot import merge_rects2
import re
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# 假设 fire_detector.py 和 utils.py 在您的环境中可用
from fire_detector import detect_fire
# from inference import YOLOInference # 在此脚本中不再需要
from utils import NmsBBoxInfo
from utils import calculate_iou, merge_rects


def extract_number(filename):
    # 此函数在单个文件测试中不是必需的，但为保持一致性而保留
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0


if __name__ == "__main__":
    # --- 单个文件测试配置 ---
    # 请在此处设置您的输入图像、检测结果txt文件和输出文件的路径
    IMAGE_PATH = "/home/manu/nfs/visi_1757316682/visi_1757316682_01005.png"
    DETECTION_TXT_PATH = "/home/manu/tmp/detections_cache_v1/visi_1757316682_01005.txt"
    OUTPUT_FILE = "/home/manu/tmp/single_output_py.txt"

    # --- 从原始脚本保留的初始化代码 ---
    # visi_1757316682 的 ROI 配置
    std_pts = np.array([
        (572, 150),
        (648, 147),
        (568, 162),
        (651, 160),
    ])
    W, H = 1280, 720
    # 根据标准点计算扩展后的ROI区域
    sx1, sy1 = np.min(std_pts, axis=0)
    sx2, sy2 = np.max(std_pts, axis=0)
    sw, sh = sx2 - sx1, sy2 - sy1
    expand_exclude_ratio = 0.5
    sx1 = int(max(0, sx1 - sw * expand_exclude_ratio))
    sy1 = int(max(0, sy1 - sh * expand_exclude_ratio))
    sw = int(min(sw * (1 + 2 * expand_exclude_ratio), W - sx1))
    sh = int(min(sh * (1 + 2 * expand_exclude_ratio), H - sy1))
    sx2 = sx1 + sw
    sy2 = sy1 + sh
    ext_xxyy = sx1, sx2, sy1, sy2

    # 初始化 detect_fire 函数所需的状态变量
    pre_fire_boxes = []
    multiFrameSwitch = True
    CheckingFireInformationGlobal = deque()

    # --- 针对单个文件的处理逻辑 ---

    # 1. 读取图像文件
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"错误: 无法在路径 {IMAGE_PATH} 找到或打开图像")

    # 2. 从TXT文件加载检测结果
    res = []
    detection_file = Path(DETECTION_TXT_PATH)
    if detection_file.exists() and detection_file.stat().st_size > 0:
        try:
            loaded_res = np.loadtxt(str(detection_file))
            # 处理文件中只有单个检测结果（1D数组）的情况
            if loaded_res.ndim == 1:
                res = [loaded_res.tolist()]
            else:
                res = loaded_res.tolist()
        except Exception as e:
            print(f"警告: 加载检测文件 {detection_file} 时出错: {e}")
    elif not detection_file.exists():
        print(f"警告: 未找到检测文件: {detection_file}")

    # 3. 将加载的检测结果转换为 NmsBBoxInfo 格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fire_results = []
    for det in res:
        x1, y1, x2, y2 = det[:4]
        w = x2 - x1
        h = y2 - y1
        box = [x1, y1, w, h]
        box = [float(b_val) for b_val in box]
        fire_result = NmsBBoxInfo(
            score=det[4],
            classID=int(det[5]),
            box=tuple(box),
        )
        fire_results.append(fire_result)

    # 4. 调用核心的火焰检测逻辑
    std_coord = (sx1, sy1, sx2, sy2)
    img_idx = 0  # 对于单张图片，索引可以设为0
    warning_boxes, pre_fire_boxes, CheckingFireInformationGlobal, filter_result, im4, log_str = detect_fire(
        fire_results,
        img_rgb,
        pre_fire_boxes,
        multiFrameSwitch,
        CheckingFireInformationGlobal,
        std_coord,
        calculate_iou,
        merge_rects,
        path_idx=img_idx
    )

    # 5. 打印结果并写入输出文件
    print("---------")
    print(f"图像: {Path(IMAGE_PATH).name}")
    print(f"检测结果: {warning_boxes}")
    print(f"log_str --> {log_str}")
    print("---------")

    with open(OUTPUT_FILE, 'w') as f_out:
        f_out.write(f"{img_idx}\t{Path(IMAGE_PATH).name}\t{log_str}\n")

    print(f"结果已成功保存到: {OUTPUT_FILE}")
