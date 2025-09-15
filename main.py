import os
# from pathlib import Path
import cv2
from collections import deque
import numpy as np
from inference import YOLOInference
from utils import Fire,NmsBBoxInfo
from fire_detector import detect_fire
from utils import calculate_iou, merge_rects
# from fire_loc_spot import merge_rects2
import time


def std_cal(std_pts):
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
    std_coord = sx1, sx2, sy1, sy2
    return std_coord

if __name__ == "__main__" :
    # img_folder = Path("./05/p")
    # img_folder = Path(r"\\172.20.254.27\青鸟消防智慧可视化02\00部门共享\【临时文件交换目录】\【to】胡靖\0912-data\0909\2p")
    # img_folder = Path(r"G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict75\2p")
    model_weight = "./models/s37e13best.onnx"
    
    model_inference = YOLOInference(weights=model_weight, conf_thresh=0.3, iou_thresh=0.45)
    std_pts = np.array([
        (828, 310), (885, 310), (945, 310),
        (826, 320), (886, 319), (946, 318),
        (826, 330), (886, 328), (948, 329)
    ])
    # std_pts = np.array([
    #     (1000,540), (1150, 630),
    # ])
    W, H = 1920, 1080
    # # Compute inclusive ROI around the standard points, expanded by a ratio
    # sx1, sy1 = np.min(std_pts, axis=0)
    # sx2, sy2 = np.max(std_pts, axis=0)
    # sw, sh = sx2 - sx1, sy2 - sy1
    # expand_exclude_ratio = 0.5
    # sx1 = int(max(0, sx1 - sw * expand_exclude_ratio))
    # sy1 = int(max(0, sy1 - sh * expand_exclude_ratio))
    # sw = int(min(sw * (1 + 2 * expand_exclude_ratio), W - sx1))
    # sh = int(min(sh * (1 + 2 * expand_exclude_ratio), H - sy1))
    # sx2 = sx1 + sw
    # sy2 = sy1 + sh
    # std_coord = sx1, sx2, sy1, sy2

    merged_labels = []
    
    pre_fire_boxes = []  # 初始化一个空
    multiFrameSwitch = True
    CheckingFireInformationGlobal = deque()

    video_path = r'G:\Windows\sxs_20250909\3\visi_1757397250.mp4'
    video_path = r'G:\Windows\20250901_stdlfire\20250901-pt5-visi.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    save_path = r'./saves'
    os.makedirs(save_path, exist_ok=True)
    write_fps = 10
    processing_interval = 15

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频尺寸: {width}x{height}, FPS: {fps}")
    vcap = None
    img_idx = 0
    # for img_idx, img_file in enumerate(img_folder.iterdir()):
    #     img_bgr = cv2.imread(str(img_file))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频处理完成")
            break

        img_idx += 1
        if img_idx % processing_interval != 0:
            continue

        if vcap is None and save_path is not None:
            vcap = cv2.VideoWriter(os.path.join(save_path, os.path.basename(video_path[:-4]) + '.mp4'),
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   write_fps, (frame.shape[1], frame.shape[0]), isColor=True)

        # img_bgr = cv2.imread(str(img_file))
        img_bgr = frame
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        res = model_inference.runs(img_bgr)
        fire_results = []
        for det in res:
            x1,y1,x2,y2 = det[:4]
            w = x2-x1
            h = y2-y1
            box = [x1,y1,w,h]
            box = [float(i) for i in box]
            fire_result = NmsBBoxInfo(score=det[4], classID=int(det[5]),box=tuple(box),)
            fire_results.append(fire_result)


        std_coord = std_cal(std_pts)
        warning_boxes, pre_fire_boxes, CheckingFireInformationGlobal, filter_result, im4 = detect_fire(fire_results,
                                                                                                  img_rgb,
                                                                                                  pre_fire_boxes,
                                                                                                  multiFrameSwitch,
                                                                                                  CheckingFireInformationGlobal,
                                                                                                  std_coord,
                                                                                                  calculate_iou,
                                                                                                  merge_rects, path_idx=img_idx
                                                                                                  )

        # print(filter_result)
        # print(pre_fire_boxes)
        # print(warning_boxes)

        # im_bbox_show = img_bgr.copy()
        # for ml in merged_labels:
        #     x1, y1, x2, y2 = ml
        #     cv2.rectangle(im_bbox_show, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2), (200, 0, 0), 1)  # blue
        im_pre_show = img_bgr.copy()

        cv2.putText(im_pre_show, f"{img_idx}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 250, 255), 2)
        for fire_obj in pre_fire_boxes:
            x1, y1, w, h = np.int_(fire_obj.fire_box)
            x2, y2 = x1 + w, y1 + h
            score = fire_obj.score
            match_flag = fire_obj.matched
            cv2.rectangle(im_pre_show, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2),
                          (0, 200, 0) if fire_obj.alarm_flag else (200, 200, 200)
                          , 1)  # green/red
            cv2.putText(im_pre_show, f"time=[{score>0.5}]{score:.2f}", (x2, y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 200, 0) if score > 0.5 else (255, 255, 255)
                        , 1)
            cv2.putText(im_pre_show, f"model=[{match_flag}]", (x2, y2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 200, 0) if match_flag else (255, 255, 255),
                        1)
            cv2.putText(im_pre_show, f"coord=[{fire_obj.queue_valid_flag}]({fire_obj.non_zero_num, fire_obj.non_outlier_num})", (x2, y2+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 200, 0) if fire_obj.queue_valid_flag else (255, 255, 255),
                        1)
            cv2.circle(im_pre_show, np.int_(fire_obj.point_queue[-1][1]), int(1), (0, 0, 200), -1)  # current_coord, blue
            cv2.circle(im_pre_show, np.int_(fire_obj.center_point), int(1), (200, 0, 200), -1)  # over_all_coord, green
        # cv2.imshow('ori', img_bgr)
        # cv2.imshow('bbox', im_bbox_show)
        if im4 is not None and True:
            im4 = im4[:im_pre_show.shape[0],:im_pre_show.shape[1],:]
            im_pre_show[-im4.shape[0]:, :im4.shape[1], :] = im4.copy()
        vcap.write(im_pre_show)
        # 显示结果
        cv2.imshow('pr', im_pre_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(0.5)
        # cv2.waitKey()

    vcap.release()
    print(f"Video {os.path.join(save_path, os.path.basename(video_path[:-4]) + '.mp4')} is saved")

    cap.release()
    cv2.destroyAllWindows()

    
