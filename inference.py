import os 
from pathlib import Path
import sys

BaseDir = str(Path(__file__).resolve().parent.parent)
sys.path.append(BaseDir)   

# from torch import nn
import numpy as np
# from einops import rearrange

import cv2 
import time

from model_common import ONNXModel



class YOLOInference():
    def __init__(self, weights=None,conf_thresh=0.25, iou_thresh=0.7):
        # super(YOLOInference, self).__init__()
        self.model = ONNXModel(weights)
        self.det_size = [self.model.input_shapes[0][2],self.model.input_shapes[0][3],self.model.input_shapes[0][1]] 
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # img_size=(长, 宽)
        # self.imgsz = img_size

    def preProcess(self, img):
        dh,dw,dc = self.det_size
        mh,mw,mc = img.shape
        assert dc == mc , "image channel not match"
        mr = mh / mw
        dr = dh / dw
        if mr > dr:
            nh = dh
            nw = int(dh / mr)
        else:
            nw = dw
            nh = int(dw * mr)
        det_scale =  float(nh / mh) # 据此可将det_size 映射到原图
        resized_img = cv2.resize(img, (nw, nh))
        det_img = np.zeros(self.det_size, dtype=np.uint8)
        det_img[:nh, :nw, :] = resized_img   # 填充到det_size尺寸


        det_inp = np.stack([det_img])

        det_inp = det_inp[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        det_inp = np.ascontiguousarray(det_inp)  # contiguous
        det_inp = det_inp.astype(np.float32)
        det_inp /= 255
        return det_inp,det_scale
        
    def postProcess(self, preds):
        preds = self.non_max_suppression(
            preds,
            self.conf_thresh,
            self.iou_thresh
        )

        results = []
        for pred in preds:
            results.append(pred)
        return results

    def forward(self, inp):
        oup = self.model.forward(inp)
        return oup
    def runs(self, img):
        # img = cv2.resize(img, self.imgsz)
        det_inp,det_scale = self.preProcess(img)
        # print(det_inp.shape,det_scale)
        oup = self.forward(det_inp)
        pred = self.postProcess(oup)[0]
        pred[:,:4] /= det_scale
        return pred
    @staticmethod
    def numpy_nms(boxes, scores, iou_threshold):
        """Pure Python NMS baseline."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.3,
                            iou_thres=0.45,  # 0.45
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nc=0,  # number of classes (optional)
                            max_time_img=0.05,
                            max_nms=30000,
                            max_wh=7680,
                            in_place=True,
                            rotated=False):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
            list of detections, on (n,6) array per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(prediction,
                      (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        if classes is not None:
            classes = np.array(classes)

        if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
            output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
            if classes is not None:
                output = [pred[np.any(pred[:, 5:6] == classes, axis=1)] for pred in output]
            return output

        bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4  # number of masks
        mi = 4 + nc  # mask start index
        xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        
        prediction = np.transpose(prediction, (0, 2, 1))  # shape(1,84,6300) to shape(1,6300,84)
        
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        t = time.time()
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]) and not rotated:
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 4))
                v[:, :4] = self.xywh2xyxy(lb[:, 1:5])  # box
                v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
                x = np.concatenate((x, v), axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = np.split(x, [4, 4 + nc], axis=1)

            if multi_label:
                i, j = np.where(cls > conf_thres)
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
            else:  # best class only
                conf = np.amax(cls, axis=1, keepdims=True)
                j = np.argmax(cls, axis=1, keepdims=True)
                x = np.concatenate((box, conf, j.astype(float), mask), axis=1)
                x = x[conf.reshape(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[np.any(x[:, 5:6] == classes, axis=1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = self.numpy_nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                # LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded

        return output
    @staticmethod
    def xywh2xyxy(x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = np.empty_like(x)  # faster than clone/copy
        xy = x[..., :2]  # centers
        wh = x[..., 2:] / 2  # half width-height
        y[..., :2] = xy - wh  # top left xy
        y[..., 2:] = xy + wh  # bottom right xy
        return y
import glob


if __name__ == "__main__":
    pt = r"./models/s37e13best.onnx"
    print(pt)
    model = YOLOInference(weights=pt, conf_thresh=0.3, iou_thresh=0.45)

    # detect_list = glob.glob(r'G:\Windows\PycharmProjects1\bot\data\b_smoker\labeling\imgs\*')
    detect_list = [r"E:\Desktop\windows\sfex6\05\p\visi_1757316682_825.jpg"]
    for imp in detect_list:
        img = cv2.imread(imp)
        if img is None:
            print(f"{imp} is None")
            continue
        results = model.runs(img)  # save predictions as labels
        print_str = '['
        for res_idx, res in enumerate(results):
            x1,y1,x2,y2,conf,cls=res
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2),
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), thickness=1)
            if res_idx != 0:
                print_str += '\n '
            print_str += f'[     {res[0]:.2f}      {res[1]:.2f}      {res[2]:.2f}      {res[3]:.2f}      {res[4]:.4f}      {res[5]}]'
        print_str += ']'
        print(print_str)
        if len(results) == 0:
            print('[]')
        cv2.imshow('Image with Rectangle', img)
        cv2.waitKey(0)
        # print(results)
        # print(results[0])
        # print(results[0].boxes.data.detach().cpu().numpy())

    # img_path = Path("../data/positive/1.jpg")
    # weights = Path("./yolov5s_fire_detection_20240822.onnx")
    # classes_name = ["fire","candle_flame","round_fire"]
    # conf_thre = 0.25
    # inference = YOLOv5Inference(weights=weights,classes_name=classes_name,conf_thre=conf_thre)
    #
    # img = cv2.imread(str(img_path))
    # pred = inference.runs(img)
    #
    # print(type(pred))
    #
    # for i, det in enumerate(pred):
    #
    #     box = np.array([int(max(0, x)) for x in det[:4]])
    #     score = round(float(det[4]),2)
    #     class_name = inference.classes_name[int(det[-1])]
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #     cv2.putText(img, f"{score} {class_name}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 0), 1)
    # cv2.imshow('Image with Rectangle and Text', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    #

    
    

