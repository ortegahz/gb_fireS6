# 描述: 实现检测框的过滤逻辑，从 vfd_ipu_detection.cpp 和 fire_detect.cpp 翻译而来。
from utils import *

def _filter_boxes(boxes: list, condition_func: callable) -> list:
    """ 一个通用的过滤框架，用于处理C++中复杂的循环和迭代器失效问题 """
    items = boxes[:]
    while True:
        removed_in_pass = False
        i = 0
        while i < len(items):
            j = 0
            while j < len(items):
                if i == j:
                    j += 1
                    continue
                
                # 如果条件函数返回True，则删除j并重启
                if condition_func(items[i], items[j]):
                    items.pop(j)
                    removed_in_pass = True
                    if j < i: i -= 1
                    # C++的逻辑是在删除后中断两个循环，然后再次从头开始
                    # 在这里，我们中断内层对j的循环，i的循环会继续
                    # 并在外层while循环中重新开始
                    break
                else:
                    j += 1
            else: # 如果内层循环正常结束（没有break）
                i += 1
                continue
            # 如果内层循环被break了
            break
        if not removed_in_pass:
            break
    return items
 
def filter_iou(fire_list: List[Fire]) -> List[Fire]:
    # print(f"in iou filter, initial size {len(fire_list)}")
    
    def condition(item_i, item_j):
        # C++逻辑: iter_j->fire_box == intersection || iou > 0.0
        # 这意味着如果j被i包含，或者它们有任何重叠，就删除j
        intersection = rect_intersection(item_i.fire_box, item_j.fire_box)
        union_area = rect_area(rect_union(item_i.fire_box, item_j.fire_box))
        
        # 修正逻辑：C++中是iter_j被包含在iter_i中
        is_contained = intersection == item_j.fire_box and rect_area(intersection) > 0
        iou = rect_area(intersection) / union_area if union_area > 0 else 0
        
        if is_contained or iou > 0.0:
            return True
        return False
 
    return _filter_boxes(fire_list, condition)
 
# def filter_low_fire(detect_boxes: List[NmsBBoxInfo]) -> List[NmsBBoxInfo]:
#     # print(f"in low fire filter, initial size {len(detect_boxes)}")
#
#     def condition(box_i, box_j):
#         # 如果i在j的正上方且距离很近，则删除j
#         i_ymax, i_xmin = box_i.box[1] + box_i.box[3], box_i.box[0]
#         j_ymin, j_xmin = box_j.box[1], box_j.box[0]
#         return abs(i_ymax - j_ymin) < 20 and abs(i_xmin - j_xmin) < 20
#
#     return _filter_boxes(detect_boxes, condition)
 
def filter_firein_tungsten(detect_boxes: List[NmsBBoxInfo]):
    fires = [b for b in detect_boxes if b.classID == 0]
    tungstens = [NmsBBoxInfo(b.score, 3, b.box) for b in detect_boxes if b.classID != 0]
 
    filtered_fires = []
    for fire in fires:
        is_inside = False
        for tungsten in tungstens:
            union_area = rect_area(rect_union(fire.box, tungsten.box))
            if union_area > 0:
                iou = rect_area(rect_intersection(fire.box, tungsten.box)) / union_area
                if iou > 0.001:
                    is_inside = True
                    break
        if not is_inside:
            filtered_fires.append(fire)
            
    return filtered_fires, tungstens