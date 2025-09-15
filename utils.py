# 描述: 定义项目中使用的数据结构和辅助函数。
from dataclasses import dataclass, field
import sys
from typing import Tuple, List, Deque
import numpy as np

@dataclass
class NmsBBoxInfo:
    """ 对应于C++的 NmsBBoxInfo 结构体 """
    score: float
    classID: int
    box: Tuple[float, float, float, float]  # (x, y, w, h)

@dataclass
class Fire:
    """ 对应于C++的 Fire 结构体 """
    fire_box: Tuple[float, float, float, float]  # (x, y, w, h)
    score: float = 0.0
    fire_point: Tuple[float, float] = (0.0, 0.0)    # 世界坐标
    center_point: Tuple[float, float] = (0.0, 0.0)  # 图像坐标
    matched: bool = True,
    point_queue: List[Tuple[float, List[float]]] = None
    queue_valid_flag: bool = False,
    alarm_flag: bool = False,
    non_zero_num: int = 0,
    non_outlier_num: int = 0,
 
@dataclass
class CheckingFireInformation:
    """ 对应于C++的 CheckingFireInformation 结构体 """
    fire_point: Tuple[float, float]
    fireShape: str
    score: float
 
@dataclass
class FireAnalysisResult:
    """ 对应于C++的 FireAnalysisResult 结构体 """
    final_x: int = 0
    final_y: int = 0
    score: float = 0.0
    shape_type: str = "others"
    span_list: List[int] = field(default_factory=list)
    sid: int = 0
    shape_specific_id: int = 0
    is_valid: bool = False
 
def rect_intersection(rect1: Tuple, rect2: Tuple) -> Tuple:
    """ 计算两个矩形的交集 """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
 
    if x_right < x_left or y_bottom < y_top:
        return (0, 0, 0, 0)
 
    w = x_right - x_left
    h = y_bottom - y_top
    return (x_left, y_top, w, h)
 
def rect_union(rect1: Tuple, rect2: Tuple) -> Tuple:
    """ 计算两个矩形的并集（包含两者的边界框） """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    x_left = min(x1, x2)
    y_top = min(y1, y2)
    x_right = max(x1 + w1, x2 + w2)
    y_bottom = max(y1 + h1, y2 + h2)
    
    w = x_right - x_left
    h = y_bottom - y_top
    return (x_left, y_top, w, h)
 
def rect_area(rect: Tuple) -> float:
    """ 计算矩形的面积 """
    return rect[2] * rect[3]


def calculate_iou(box1, box2):
    """Intersection-over-Union between two boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def merge_rects(rects):
    """Greedily merge overlapping or nearby rectangles into unions."""
    merged_flag = True
    merge_count = 0
    i_max = len(rects)
    while merged_flag:
        merge_count += 1
        merged_flag = False
        new_rects = []
        newly_update_rects = []
        merged_set = set()
        for i, rect1 in enumerate(rects):
            if i in merged_set:
                continue
            if i >= i_max:
                new_rects.append(rect1)
                continue
            merged = False
            for j, rect2 in enumerate(rects):
                if i < j and j not in merged_set and i not in merged_set:
                    xa1, ya1, wa, ha = rect1.box
                    xa2, ya2 = xa1 + wa, ya1 + ha
                    xb1, yb1, wb, hb = rect2.box
                    xb2, yb2 = xb1 + wb, yb1 + hb
                    box1 = (xa1, ya1, xa2 - xa1, ya2 - ya1)
                    box2 = (xb1, yb1, xb2 - xb1, yb2 - yb1)
                    # if calculate_iou(box1, box2) != 0 or abs(yb1 - ya2) < max(4, (ya2 - ya1), yb2 - yb1) or \
                    #         abs(ya1 - yb2) < max(4, (ya2 - ya1), yb2 - yb1) or \
                    #         abs(xb1 - xa2) < max(4, (xa2 - xa1), xb2 - xb1) or abs(xa1 - xb2) < max(4, (xa2 - xa1),
                    #                                                                                 xb2 - xb1):

                    if calculate_iou(box1, box2) != 0 or \
                            ((abs(yb1 - ya2) < max(4, (ya2 - ya1), yb2 - yb1) or abs(ya1 - yb2) < max(4, (ya2 - ya1), yb2 - yb1)) and \
                             (abs(xb1 - xa2) < max(4, (xa2 - xa1), xb2 - xb1) or abs(xa1 - xb2) < max(4, (xa2 - xa1), xb2 - xb1))):
                        xcombine_1 = min(xa1, xb1)
                        xcombine_2 = max(xa2, xb2)
                        ycombine_1 = min(ya1, yb1)
                        ycombine_2 = max(ya2, yb2)
                        newly_update_rects.append(NmsBBoxInfo(max(rect1.score, rect2.score), 0, (xcombine_1, ycombine_1, xcombine_2 - xcombine_1, ycombine_2 - ycombine_1)))
                        # newly_update_rects.append((int(xcombine_1), int(ycombine_1), int(xcombine_2), int(ycombine_2)))
                        merged_set.add(i)
                        merged_set.add(j)
                        merged = True
                        merged_flag = True
                        continue
            if not merged:
                new_rects.append(rect1)
        rects = newly_update_rects
        i_max = len(rects)
        for r in new_rects:
            rects.append(r)
    return rects