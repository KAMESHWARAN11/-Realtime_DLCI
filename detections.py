import sys
import cv2
import torch
import numpy as np

# YOLOv5 utility imports
sys.path.insert(0, './yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

# -----------------------------
# LOAD MODEL
# -----------------------------
def Load_Yolo_Model(device=None, conf_thres=0.5, iou_thres=0.45,
                    weights='models/yolov5s.pt', imgsz=640, track_only=[]):

    device = select_device('') if device is None else device
    yolov5 = attempt_load(weights, map_location=device)

    yolov5.device = device
    yolov5.is_half = yolov5.device.type != 'cpu'

    if yolov5.is_half:
        yolov5.half()

    yolov5.imgsz = check_img_size(imgsz, s=yolov5.stride.max())
    yolov5.names = yolov5.module.names if hasattr(yolov5, 'module') else yolov5.names

    yolov5.conf_thres = conf_thres
    yolov5.iou_thres = iou_thres
    yolov5.classes = None if not track_only else track_only
    yolov5.agnostic_nms = False

    # 🔥 WARMUP (IMPORTANT)
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    _ = yolov5(dummy.half() if yolov5.is_half else dummy.float())

    return yolov5


# -----------------------------
# BOX FORMAT CONVERSION
# -----------------------------
def xyxy2tlwh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# -----------------------------
# YOLO PREDICTION
# -----------------------------
def yolo_predict(yolov5, img, im0s, log=None):

    boxes, class_inds, scores = [], [], []

    try:
        img = torch.from_numpy(img).to(yolov5.device)
        img = img.half() if yolov5.is_half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()

        pred = yolov5(img)[0]

        pred = non_max_suppression(
            pred,
            yolov5.conf_thres,
            yolov5.iou_thres,
            classes=yolov5.classes,
            agnostic=yolov5.agnostic_nms
        )

        t2 = time_synchronized()

        for det in pred:
            if len(det):

                # Rescale boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:

                    xyxy_tensor = torch.tensor(xyxy).view(1, 4)

                    tlwh = xyxy2tlwh(xyxy_tensor).view(-1).tolist()

                    boxes.append(tlwh)
                    class_inds.append(int(cls))
                    scores.append(float(conf))

        # 🔥 SIMPLE LOG
        if log and hasattr(log, 'info'):
            log.info(f"Detection Done ({t2 - t1:.3f}s)")
        else:
            print(f"Detection Done ({t2 - t1:.3f}s)")

    except Exception as e:
        if log and hasattr(log, 'error'):
            log.error(f"Prediction Error: {e}")
        else:
            print(f"Prediction Error: {e}")

    return boxes, class_inds, scores