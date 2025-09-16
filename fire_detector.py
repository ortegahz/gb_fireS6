# 描述: 主检测逻辑，从 fire_detect.cpp 翻译而来。

# SHAPE_WEIGHTS = {"short": 1.0, "funnel": 0.9, "fusiform": 0.4, "rectangle": 0.3, "others": 0.1}
import cv2
# from fire_shape_analysis import *
# from correction import *
import numpy as np

# from collections import deque
# from utils import *
from filters import *
from fire_loc_spot import fire_locate
from outlier_queue import outlier_filter

SHAPE_NAMES = ['圆', '葫芦', '类圆', '菱形', '长条', '其他']
SHAPE_SCORES = [0.72, 0.9, 0.7, 0.5, 0.3, 0.1]

W, H = 1920, 1080


# def get_shape_weight(shape: str) -> float:
#     return SHAPE_WEIGHTS.get(shape, SHAPE_WEIGHTS["others"])

# def calculate_weighted_average_from_inliers(points: Deque[CheckingFireInformation], threshold: float) -> Tuple[
#     float, float]:
#     if not points:
#         return 0.0, 0.0
#
#     outlier_index = -1
#     if len(points) > 2:
#         pts_array = np.array([p.fire_point for p in points])
#         centroid = np.mean(pts_array, axis=0)
#         distances_sq = np.sum((pts_array - centroid) ** 2, axis=1)
#         potential_outlier_idx = np.argmax(distances_sq)
#         if np.sqrt(distances_sq[potential_outlier_idx]) > threshold:
#             outlier_index = potential_outlier_idx
#
#     total_weighted_x, total_weighted_y, total_weight = 0.0, 0.0, 0.0
#     for i, info in enumerate(points):
#         if i == outlier_index: continue
#         weight = info.score * get_shape_weight(info.fireShape)
#         total_weighted_x += info.fire_point[0] * weight
#         total_weighted_y += info.fire_point[1] * weight
#         total_weight += weight
#
#     if total_weight == 0:
#         return (0.0, 0.0)
#     return total_weighted_x / total_weight, total_weighted_y / total_weight

def detect_fire(
        results,  #: List[NmsBBoxInfo],
        img,  #: np.ndarray,
        pre_fire_boxes,  #: List[Fire],
        multiFrameSwitch,  #: bool,
        CheckingFireInformationGlobal,  #: Deque[CheckingFireInformation],
        std_coord,
        calculate_iou,
        merge_rects,
        path_idx=0,
        queue_max_len=10
):  # -> Tuple[List[Fire], List[Fire], Deque[CheckingFireInformation]]:
    im4_ret = None
    height, width = img.shape[:2]
    filter_result, tungsten_result = filter_firein_tungsten(results)
    # before_filter_low_fire_size = len(filter_result)
    # if not before_filter_low_fire_size: CheckingFireInformationGlobal.clear()

    # filter_result = filter_low_fire(filter_result)
    # after_filter_low_fire_size = len(filter_result)

    # 滤除九点（带50%扩展）外的火焰
    responding_labels = []
    sx1, sy1, sx2, sy2 = std_coord
    std_box = (sx1, sy1, sx2 - sx1, sy2 - sy1)
    for fr in filter_result:
        if fr.classID == 0:
            # x1, y1, w, h = fr.box
            # x2, y2 = x1 + w, y1 + h
            # ori_labels.append([x1, y1, x2, y2])
            if calculate_iou(fr.box, std_box) == 0:
                continue
            # responding_labels.append(NmsBBoxInfo(fr.score, 0, fr.box))
            responding_labels.append(fr)
            # responding_labels.append([x1, y1, x2, y2])

    # 融框
    merged_labels = merge_rects(responding_labels)

    # print(f'merged_labels={merged_labels}')
    # print(f'responding_labels={[np.int_(rl.box) for rl in responding_labels]}')
    # 清理检测框使其在图像边界内
    valid_boxes = []
    for item in merged_labels:
        x, y, w, h = item.box
        if not (w <= 0 or h <= 0 or x + w <= 0 or x >= width or y + h <= 0 or y >= height):
            x, y = max(x, 0.0), max(y, 0.0)
            item.box = (x, y, min(x + w, width) - x, min(y + h, height) - y)
            valid_boxes.append(item)
    filter_result = valid_boxes
    # pre_match = [False] * len(pre_fire_boxes)

    for box in pre_fire_boxes:
        box.matched = False

    multiFrameSwitch = True  # set as constant True
    # 核心多帧处理逻辑
    if multiFrameSwitch:
        cur_detections = filter_result  # [r for r in filter_result if r.classID != 3]

        for pre_idx, pre_item in enumerate(pre_fire_boxes):
            best_match_idx, best_iou = -1, 0.0
            for cur_idx, cur_item in enumerate(cur_detections):
                union_area = rect_area(rect_union(pre_item.fire_box, cur_item.box))
                iou = rect_area(
                    rect_intersection(pre_item.fire_box, cur_item.box)) / union_area if union_area > 0 else 0
                if iou > best_iou: best_iou, best_match_idx = iou, cur_idx

            if best_iou > 0.0:
                cur_item = cur_detections.pop(best_match_idx)
                pre_item.score = min(pre_item.score + max(0.18, (cur_item.score - 0.25) * 0.8), 1.0)
                pre_item.fire_box = cur_item.box
                pre_item.matched = True

                # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # x1, y1, w, h = cur_item.box
                # x2, y2 = x1 + w, y1 + h
                # ext_xxyy = sx1, sx2, sy1, sy2
                # shape_id_best, coord_best, weight_best, im_best = fire_locate(img_bgr, (x1, y1, x2, y2),
                #                                                               global_offset_x=0, global_offset_y=0,
                #                                                               ext_xxyy=ext_xxyy, ret_all=False, path_idx=path_idx)
                # if im_best is not None:
                #     cv2.imshow(f"fire{pre_idx}", im_best)
                # pre_item.center_point = coord_best
                # ... 火点校正逻辑 ...

        pre_fire_boxes = [box for i, box in enumerate(pre_fire_boxes) if box.matched or box.score - 0.05 >= 1e-6]
        for i, box in enumerate(pre_fire_boxes):
            if not box.matched: box.score -= 0.05
    else:
        pre_fire_boxes.clear()
    # 将新的检测添加到 pre_fire_boxes
    for item in filter_result:
        # if item.classID == 3: continue
        score = max(0.15, (item.score - 0.25) / 2.0) if multiFrameSwitch else 0.6
        new_fire = Fire(fire_box=item.box, score=score)
        pre_fire_boxes.append(new_fire)
    log_str = ''
    log_str += f'--- Per-Fire Analysis (Total {len(pre_fire_boxes)} fires) ---\n'
    for pre_idx, pre_item in enumerate(pre_fire_boxes):
        log_str += f'\n[Fire Object {pre_idx}] Box: {pre_item.fire_box}, Score: {pre_item.score:.3f}, Matched: {pre_item.matched}\n'
        # 火点坐标计算
        if pre_item.matched:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            x1, y1, w, h = pre_item.fire_box
            x2, y2 = x1 + w, y1 + h
            ext_xxyy = sx1, sx2, sy1, sy2
            shape_id_best, coord_best, weight_best, im_best = fire_locate(img_bgr, (x1, y1, x2, y2), global_offset_x=0,
                                                                          global_offset_y=0, ext_xxyy=ext_xxyy,
                                                                          ret_all=False, path_idx=path_idx)
            if im_best is not None:
                im4_ret = im_best
            log_str += f'  - fire_locate (matched): coord={coord_best}, weight={weight_best:.3f}, shape_id={shape_id_best}\n'
        else:
            weight_best, coord_best = 0.0, (0.0, 0.0)
            im_best, shape_id_best = None, None
            log_str += f'  - fire_locate (unmatched): placeholder coord=(0.0, 0.0), weight=0.0, shape_id=None\n'
        if type(coord_best) == np.ndarray: coord_best = list(coord_best)
        if pre_item.point_queue is None:
            pre_item.point_queue = []
        pre_item.point_queue.append((weight_best, coord_best))
        if len(pre_item.point_queue) > queue_max_len:
            pre_item.point_queue = pre_item.point_queue[-queue_max_len:]
        log_str += f'{path_idx},{pre_idx},{pre_item.fire_box},{pre_item.point_queue}\n'
        print(f'{path_idx}: F{pre_idx}={pre_item.point_queue}')
        queue_valid_flag, center_point, non_zero_num, non_outlier_num = outlier_filter(pre_item.point_queue,
                                                                                       min_valid_num=(5, 4))
        log_str += f'  - outlier_filter result: valid_flag={queue_valid_flag}, center_point={center_point}, non_zero_num={non_zero_num}, non_outlier_num={non_outlier_num}\n'
        pre_item.center_point = center_point
        pre_item.queue_valid_flag = queue_valid_flag
        pre_item.non_zero_num = non_zero_num
        pre_item.non_outlier_num = non_outlier_num
        pre_item.alarm_flag = queue_valid_flag and pre_item.score > 0.5
        log_str += f'  - Alarm Flag Calculation: queue_valid_flag({queue_valid_flag}) AND score({pre_item.score:.3f}) > 0.5 (is {pre_item.score > 0.5}) ==> alarm_flag={pre_item.alarm_flag}\n'

        # pre_fire_boxes.append(new_fire)
        if im_best is not None:
            cv2.imshow(f"fire{len(pre_fire_boxes) - 1}", im_best)

    # 根据钨丝灯位置惩罚分数
    tungstens = tungsten_result
    log_str += '\n--- Final Adjustments ---\n[Tungsten Penalty]\n'
    for fire in pre_fire_boxes:
        for t_box in tungstens:
            union_area = rect_area(rect_union(fire.fire_box, t_box.box))
            if union_area > 0 and rect_area(rect_intersection(fire.fire_box, t_box.box)) / union_area >= 0.001:
                log_str += f'  - Fire at {fire.fire_box} penalized. Score {fire.score:.3f} -> {fire.score - 0.5:.3f}.\n'
                fire.score -= 0.5
                break

    log_str += '[Final Filtering]\n'
    log_str += f'  - Before score-based filtering: {len(pre_fire_boxes)} fires.\n'
    pre_fire_boxes = [box for box in pre_fire_boxes if box.score >= 1e-6]
    log_str += f'  - After score-based filtering (score >= 1e-6): {len(pre_fire_boxes)} fires remain.\n'
    log_str += f'  - Before IOU-based filtering: {len(pre_fire_boxes)} fires.\n'
    pre_fire_boxes = filter_iou(pre_fire_boxes)
    log_str += f'  - After IOU-based filtering: {len(pre_fire_boxes)} fires remain.\n'

    warning_boxes = [box for box in pre_fire_boxes if box.alarm_flag]
    log_str += f'\n--- Result ---\nFound {len(warning_boxes)} warning boxes.\n'
    # print(f"warning fire size = {len(warning_boxes)}")

    return warning_boxes, pre_fire_boxes, CheckingFireInformationGlobal, filter_result, im4_ret, log_str
