"""YOLOv8n ONNX detector: load / preprocess / postprocess / NMS.
Supports both detection (1 output) and segmentation (2 outputs) models.
Each detection is returned as (cls_id, conf, (x1,y1,x2,y2), mask|None)."""
import os

import cv2
import numpy as np
import onnxruntime as ort


def default_model(here):
    """prefer segmentation model when present (silhouette rendering)."""
    seg = os.path.join(here, "models", "yolov8n-seg.onnx")
    if os.path.exists(seg):
        return seg
    return os.path.join(here, "models", "yolov8n.onnx")

# COCO class grouping (decides coloring)
HOSTILE = {0}                                    # person
DEVICE = {62, 63, 64, 65, 66, 67, 68, 69, 70, 72}  # tv, laptop, mouse, remote,
#        keyboard, cell phone, microwave, oven, toaster, refrigerator
VEHICLE = {1, 2, 3, 4, 5, 6, 7, 8}               # bicycle..boat
INFRA = {9, 10, 11, 12}                          # traffic light, fire hydrant,
#        stop sign, parking meter  (street furniture = hackable in-game)
ANIMAL = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}  # bird..giraffe

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
          "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
          "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"]


def category(cls_id: int) -> str:
    if cls_id in HOSTILE:
        return "hostile"
    if cls_id in DEVICE:
        return "device"
    if cls_id in VEHICLE:
        return "vehicle"
    if cls_id in INFRA:
        return "infra"
    if cls_id in ANIMAL:
        return "animal"
    return "other"


class Detector:
    def __init__(self, model_path: str, input_size: int = 640, conf: float = 0.35,
                 nms_iou: float = 0.45):
        self.size = input_size
        self.conf = conf
        self.nms_iou = nms_iou
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(4, min(8, (os.cpu_count() or 4)))
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        outs = self.sess.get_outputs()
        shapes = [(o.name, o.shape) for o in outs]
        if len(outs) == 1:
            self.mode = "det"
            self.proto_hw = None
        elif len(outs) == 2:
            self.mode = "seg"                    # decoded (1,116,8400) + proto
            self.proto_hw = outs[1].shape[2:]
        else:
            # raw multi-scale export: proto(1,32,160,160) +
            # per-scale concat(1,144,H,W)[=64 dfl + 80 cls] + mask coeff(1,32,H,W)
            self.mode = "raw"
            self.raw_scales = []                 # (stride, concat_idx, coeff_idx)
            self.proto_idx = None
            for i, (_, sh) in enumerate(shapes):
                c, h, w = sh[1], sh[2], sh[3]
                if c == 32 and h == 160:
                    self.proto_idx = i
                elif c == 144:
                    stride = self.size // h
                    # matching 32-ch coeff output at same resolution
                    ci = next(j for j, (_, s2) in enumerate(shapes)
                              if s2[1] == 32 and s2[2] == h)
                    self.raw_scales.append((stride, i, ci))
            self.raw_scales.sort()
            self.proto_hw = (160, 160)
            self._project = np.arange(16, dtype=np.float32)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = min(self.size / w, self.size / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, s, px, py

    def _mask(self, coeff, box640, proto_r, s, px, py, w0, h0):
        """decode one instance mask into full-frame uint8 mask (0/255)."""
        mh, mw = self.proto_hw
        m = coeff @ proto_r                          # (mh*mw,)
        m = 1.0 / (1.0 + np.exp(-m)).reshape(mh, mw)
        m = cv2.resize(m, (self.size, self.size))    # to letterbox coords
        x1, y1, x2, y2 = [int(round(v)) for v in box640]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.size, x2), min(self.size, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = m[y1:y2, x1:x2]
        ox1, oy1 = max(0, int((x1 - px) / s)), max(0, int((y1 - py) / s))
        ox2, oy2 = min(w0, int((x2 - px) / s)), min(h0, int((y2 - py) / s))
        bw, bh = ox2 - ox1, oy2 - oy1
        if bw < 3 or bh < 3 or crop.size == 0:
            return None
        crop = cv2.resize(crop, (bw, bh))
        full = np.zeros((h0, w0), np.uint8)
        full[oy1:oy2, ox1:ox2] = (crop > 0.5).astype(np.uint8) * 255
        return full

    def _decode_raw(self, outs):
        """decode raw multi-scale outputs -> (boxes cxcywh@640, confs, clss, coeffs)"""
        boxes_l, confs_l, clss_l, coeffs_l = [], [], [], []
        for stride, ci, mi in self.raw_scales:
            concat = outs[ci][0]                       # (144, H, W)
            _, H, W = concat.shape
            n = H * W
            box = concat[:64].reshape(4, 16, n)
            # softmax over 16 reg bins -> distance expectation
            e = np.exp(box - box.max(axis=1, keepdims=True))
            dist = ((e / e.sum(axis=1, keepdims=True))
                    * self._project[None, :, None]).sum(axis=1)   # (4, n)
            logits = concat[64:].reshape(80, n)        # (80, n)
            scores = 1.0 / (1.0 + np.exp(-logits))
            coeff = outs[mi][0].reshape(32, n).T       # (n, 32)
            gy, gx = np.meshgrid(np.arange(H, dtype=np.float32),
                                 np.arange(W, dtype=np.float32), indexing="ij")
            gx, gy = gx.reshape(-1), gy.reshape(-1)
            l, t, r, b = dist[0], dist[1], dist[2], dist[3]
            cx = (gx + 0.5 - l) * stride
            cy = (gy + 0.5 - t) * stride
            bw = (l + r) * stride
            bh = (t + b) * stride
            boxes_l.append(np.stack([cx, cy, bw, bh], axis=1))
            confs_l.append(scores.max(axis=0))
            clss_l.append(scores.argmax(axis=0))
            coeffs_l.append(coeff)
        boxes = np.concatenate(boxes_l)
        confs = np.concatenate(confs_l)
        clss = np.concatenate(clss_l)
        coeffs = np.concatenate(coeffs_l)
        keep = confs > self.conf
        return boxes[keep], confs[keep], clss[keep], coeffs[keep]

    def __call__(self, frame):
        """return list of (cls_id, conf, (x1,y1,x2,y2), mask|None)"""
        h0, w0 = frame.shape[:2]
        canvas, s, px, py = self._letterbox(frame)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = self.sess.run(None, {self.input_name: blob})

        if self.mode == "raw":
            boxes, confs, clss, coeffs = self._decode_raw(outs)
            proto_r = outs[self.proto_idx][0].reshape(32, -1)
        else:
            pred = outs[0][0].T                            # (8400, 84|116)
            scores = pred[:, 4:84] if self.mode == "seg" else pred[:, 4:]
            clss = scores.argmax(1)
            confs = scores.max(1)
            keep = confs > self.conf
            boxes = pred[keep, :4].copy()                  # cx, cy, w, h (640 coords)
            confs = confs[keep]
            clss = clss[keep]
            coeffs = pred[keep, 84:] if self.mode == "seg" else None
            if self.mode == "seg":
                proto_r = outs[1][0].reshape(outs[1][0].shape[0], -1)
            else:
                proto_r = None
        if not len(boxes):
            return []
        # cxcywh -> xyxy
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        idxs = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(),
                                         self.conf, self.nms_iou)).flatten()
        out = []
        for i in idxs:
            x1, y1, x2, y2 = boxes[i]
            # un-letterbox back to original frame coordinates
            x1 = (x1 - px) / s
            y1 = (y1 - py) / s
            x2 = (x2 - px) / s
            y2 = (y2 - py) / s
            x1, x2 = sorted((max(0, x1), min(w0 - 1, x2)))
            y1, y2 = sorted((max(0, y1), min(h0 - 1, y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            mask = None
            if proto_r is not None:
                mask = self._mask(coeffs[i], boxes[i], proto_r, s, px, py, w0, h0)
            out.append((int(clss[i]), float(confs[i]), (x1, y1, x2, y2), mask))
        return out
