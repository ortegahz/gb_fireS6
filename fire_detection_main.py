# from fire_loc_spot import merge_rects2
import glob
import os
import re
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from fire_detector import detect_fire
from inference import YOLOInference
from utils import NmsBBoxInfo
from utils import calculate_iou, merge_rects


def extract_number(filename):
    # 使用正则表达式找出最后一个 _ 和 . 之间的数字
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return 0  # 如果没有找到，则返回0作为默认值


if __name__ == "__main__":
    # --- 功能开关和配置 ---
    # 设置为 True: 运行YOLO模型进行推理并保存检测结果。
    # 设置为 False: 从本地加载已保存的检测结果，跳过模型推理。
    RUN_YOLO_INFERENCE = True
    DETECTION_CACHE_DIR = "/home/manu/tmp/detections_cache"
    OUTPUT_FILE = "/home/manu/tmp/output_gb_s6_py.txt"

    # img_folder = Path("./05/p")
    # img_folder = Path(r"\\172.20.254.27\青鸟消防智慧可视化02\00部门共享\【临时文件交换目录】\【to】胡靖\0912-data\0909\2p")
    # img_folder = Path(r"G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict75\2p")
    model_inference = None
    if RUN_YOLO_INFERENCE:
        model_weight = "./models/s37e13best.onnx"
        model_inference = YOLOInference(weights=model_weight, conf_thresh=0.3, iou_thresh=0.45)
    os.makedirs(DETECTION_CACHE_DIR, exist_ok=True)
    # std_pts = np.array([
    #     (828, 310), (885, 310), (945, 310),
    #     (826, 320), (886, 319), (946, 318),
    #     (826, 330), (886, 328), (950, 331)
    # ])
    std_pts = np.array([
        (768, 291), (892, 308),
    ])
    W, H = 1920, 1080
    # Compute inclusive ROI around the standard points, expanded by a ratio
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

    merged_labels = []
    pre_fire_boxes = []  # 初始化一个空
    multiFrameSwitch = True
    CheckingFireInformationGlobal = deque()

    img_folder = glob.glob("/home/manu/nfs/visi_1757316682/*")
    sorted_file_list = sorted(img_folder, key=extract_number)
    img_idx = 0
    with open(OUTPUT_FILE, 'w') as f_out:
        for i in sorted_file_list:
            image_stem = Path(i).stem
            cache_file = Path(DETECTION_CACHE_DIR) / f"{image_stem}.txt"

            img_bgr = cv2.imread(i)
            res = []
            if RUN_YOLO_INFERENCE:
                if model_inference is None:
                    raise RuntimeError("YOLOInference模型未初始化。请设置 RUN_YOLO_INFERENCE = True")
                res = model_inference.runs(img_bgr)
                np.savetxt(cache_file, res, fmt='%.6f')
            else:
                if cache_file.exists() and cache_file.stat().st_size > 0:
                    try:
                        loaded_res = np.loadtxt(str(cache_file))
                        # 处理只有单个检测结果（1D数组）的情况
                        if loaded_res.ndim == 1:
                            res = [loaded_res.tolist()]
                        else:
                            res = loaded_res.tolist()
                    except Exception as e:
                        print(f"警告: 加载缓存文件 {cache_file} 时出错: {e}")
                elif not cache_file.exists():
                    print(f"警告: 未找到缓存文件: {cache_file}")

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

            std_coord = (sx1, sy1, sx2, sy2)
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
            print(img_idx, "---------", warning_boxes)
            # 将结果写入output.txt
            f_out.write(f"{img_idx}\t{Path(i).name}\t{warning_boxes}\n")
            img_idx += 1
