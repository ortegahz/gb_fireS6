import os
# from pathlib import Path
import cv2
from collections import deque
import numpy as np
from inference import YOLOInference
from utils import Fire,NmsBBoxInfo
from fire_detector import detect_fire
from utils import calculate_iou, merge_rects
import glob
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
    std_coord = sx1, sy1, sx2, sy2
    return std_coord

if __name__ == "__main__" :
    # img_folder = Path("./05/p")
    # img_folder = Path(r"\\172.20.254.27\青鸟消防智慧可视化02\00部门共享\【临时文件交换目录】\【to】胡靖\0912-data\0909\2p")
    # img_folder = Path(r"G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict75\2p")
    model_weight = "./models/s37e13best.onnx"
    
    model_inference = YOLOInference(weights=model_weight, conf_thresh=0.3, iou_thresh=0.45)
    std_pts_09093 = np.array([
        (858, 307), (979, 314),
        (854, 326), (983, 324)
    ])
    std_pts_09092 = np.array([
        (828, 310), (885, 310), (945, 310),
        (826, 320), (886, 319), (946, 318),
        (826, 330), (886, 328), (950, 331)
    ])
    # std_pts_09092 = np.array([
    #     (829,314), (950, 331),
    # ])
    std_pts_09091 = np.array([
        (768,291), (892, 308),
    ])

    std_pts_09085 = np.array([
        (572, 150),
        (648, 147),
        (568, 162),
        (651, 160),
    ])
    std_pts_09084 = np.array([
        (586, 160),
        (661, 161),
        (582, 174),
        (668, 175),
    ])
    std_pts_09083 = np.array([
        (608, 257),
        (687, 258),
        (606, 271),
        (693, 271),
    ])
    std_pts_0901 = np.array([
        (1004,579), (1004,601), (1117, 571), (1127, 592),
    ])
    W0, H0 = 1920, 1080
    W1, H1 = 1280, 720

    merged_labels = []
    pre_fire_boxes = []  # 初始化一个空
    multiFrameSwitch = True
    CheckingFireInformationGlobal = deque()
    video_paths = []

    video_paths2 = glob.glob(r'G:\Windows\sxs_20250909\3\visi*.mp4')
    video_paths2 = [(vp, std_pts_09093, (W0, H0)) for vp in video_paths2]
    video_paths.extend(video_paths2)
    video_paths2 = glob.glob(r'G:\Windows\sxs_20250909\2\visi*.mp4')
    video_paths2 = [(vp, std_pts_09092, (W0, H0)) for vp in video_paths2]
    video_paths.extend(video_paths2)
    video_paths2 = glob.glob(r'G:\Windows\sxs_20250909\1\visi*.mp4')
    video_paths2 = [(vp, std_pts_09091, (W0, H0)) for vp in video_paths2]
    video_paths.extend(video_paths2)

    video_paths2 = glob.glob(r'G:\Windows\sxs_20250908\5\visi*.mp4')
    video_paths2 = [(vp, std_pts_09085, (W1, H1)) for vp in video_paths2]
    video_paths.extend(video_paths2)
    video_paths2 = glob.glob(r'G:\Windows\sxs_20250908\4\visi*.mp4')
    video_paths2 = [(vp, std_pts_09084, (W1, H1)) for vp in video_paths2]
    video_paths.extend(video_paths2)
    video_paths2 = glob.glob(r'G:\Windows\sxs_20250908\2\visi*.mp4')
    video_paths2 = [(vp, std_pts_09083, (W1, H1)) for vp in video_paths2]
    video_paths.extend(video_paths2)

    video_paths2 = glob.glob(r'G:\Windows\20250901_stdlfire\*-visi.mp4')
    video_paths2 = [(vp, std_pts_0901, (W0, H0)) for vp in video_paths2]
    video_paths.extend(video_paths2)

    video_paths2 = glob.glob(r'G:\Windows\sxs_20250908\3\visi*.mp4')# 30min...
    video_paths2 = [(vp, std_pts_09083, (W1, H1)) for vp in video_paths2]
    video_paths.extend(video_paths2)
    # video_paths = glob.glob(r'G:\Windows\sxs_20250909\3\visi*250.mp4')
    # video_paths = [(vp, std_pts_09093, (W0, H0)) for vp in video_paths]
    write_log_flag = True
    time_verbose = False
    for video_idx, video_pair in enumerate(video_paths[4:]):
        pre_fire_boxes = []  # 初始化一个空
        video_path, std_pts, sz = video_pair
        W, H = sz
        print(f'V{video_idx+1}/{len(video_paths)}: {video_path}')
        # video_path = r'G:\Windows\sxs_20250909\3\visi_1757397250.mp4'
        # video_path = r'G:\Windows\20250901_stdlfire\20250901-pt5-visi.mp4'
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
            time1 = time.time()

            # if img_idx < 3750:
            #     res=np.array([])
            # else:
            #     res = model_inference.runs(img_bgr)
            # if img_idx > 3840:
            #     print(img_idx)
            res = model_inference.runs(img_bgr)

            if write_log_flag:
                video_log_name = os.path.basename(video_path)[:-4]
                save_video_log_path = os.path.join(save_path, video_log_name)
                os.makedirs(save_video_log_path, exist_ok=True)
                cv2.imwrite(os.path.join(save_video_log_path, f'{video_log_name}_{img_idx:05d}.png'), img_bgr)
                save_lines = []
                for r in res:
                    x1, y1, x2, y2, conf, cls = r
                    save_lines.append(
                        f'{int(cls)} {(x1 + x2) / 2 / 1920:.6f} {(y1 + y2) / 2 / 1080:.6f} {(x2 - x1) / 1920:.6f} {(y2 - y1) / 1920:.6f} \n')
                with open(os.path.join(save_video_log_path, f'{video_log_name}_{img_idx:05d}.txt'), 'w') as f:
                    if len(save_lines) == 0:
                        f.write('\n')
                    for sl in save_lines:
                        f.write(sl)

            time2 = time.time()
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
            warning_boxes, pre_fire_boxes, CheckingFireInformationGlobal, filter_result, im4, log_str = detect_fire(fire_results,
                                                                                                      img_rgb,
                                                                                                      pre_fire_boxes,
                                                                                                      multiFrameSwitch,
                                                                                                      CheckingFireInformationGlobal,
                                                                                                      std_coord,
                                                                                                      calculate_iou,
                                                                                                      merge_rects, path_idx=img_idx
                                                                                                      )

            if write_log_flag and len(log_str) > 0:
                with open(os.path.join(save_path, f'{video_log_name}_info.txt'), 'a') as f:
                    f.write(log_str)
            # print(filter_result)
            # print(pre_fire_boxes)
            # print(warning_boxes)

            # im_bbox_show = img_bgr.copy()
            # for ml in merged_labels:
            #     x1, y1, x2, y2 = ml
            #     cv2.rectangle(im_bbox_show, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2), (200, 0, 0), 1)  # blue
            im_pre_show = img_bgr.copy()

            if time_verbose:
                print(f'onnx:{time2-time1:.3f}s')
                print(f'proc:{time.time()-time2:.3f}s')
            cv2.putText(im_pre_show, f"{img_idx}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 250, 255), 2)
            for fire_obj_idx, fire_obj in enumerate(pre_fire_boxes):
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
                cv2.putText(im_pre_show, f"{np.int_(np.round(fire_obj.center_point))}", (10, 25*fire_obj_idx+80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255),
                            1)
                cv2.circle(im_pre_show, np.int_(np.round(fire_obj.point_queue[-1][1])), int(1), (200, 180, 0), -1)  # current_coord, blue
                cv2.circle(im_pre_show, np.int_(np.round(fire_obj.center_point)), int(1), (0, 0, 0), -1)  # over_all_coord, green
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

    
