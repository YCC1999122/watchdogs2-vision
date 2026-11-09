"""IoU tracker + EMA smoothing + velocity extrapolation + mask carry-over."""
import time

import cv2
import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    _next_id = 1

    def __init__(self, cls_id, conf, box, mask=None):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_id = cls_id
        self.conf = conf
        self.box = list(box)          # smoothed box
        self.mask_soft = None         # full-frame float 0..1, EMA-smoothed over time
        self.vel = (0.0, 0.0)         # center velocity px/frame
        self.miss = 0
        self.born = time.perf_counter()
        if mask is not None:
            self.mask_soft = mask.astype(np.float32) / 255.0

    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def predict(self):
        """extrapolate on frames without detection. The soft mask is shifted
        by the velocity so the fill stays glued to the (unseen) object instead
        of disappearing or staying frozen at the old location."""
        cx, cy = self.center()
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        dx, dy = self.vel
        nx, ny = cx + dx, cy + dy
        self.box = [nx - w / 2, ny - h / 2, nx + w / 2, ny + h / 2]
        if self.mask_soft is not None and (abs(dx) > 0.2 or abs(dy) > 0.2):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            self.mask_soft = cv2.warpAffine(
                self.mask_soft, M,
                (self.mask_soft.shape[1], self.mask_soft.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)
        self.miss += 1
        return self

    def update(self, conf, box, ema_old, mask=None):
        cx, cy = self.center()
        nx1, ny1, nx2, ny2 = box
        ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        self.vel = (0.6 * self.vel[0] + 0.4 * (ncx - cx),
                    0.6 * self.vel[1] + 0.4 * (ncy - cy))
        o, n = ema_old, 1.0 - ema_old
        self.box = [o * self.box[0] + n * nx1, o * self.box[1] + n * ny1,
                    o * self.box[2] + n * nx2, o * self.box[3] + n * ny2]
        self.conf = conf
        if mask is not None:
            new_soft = mask.astype(np.float32) / 255.0
            if self.mask_soft is None or self.mask_soft.shape != new_soft.shape:
                self.mask_soft = new_soft
            else:
                self.mask_soft = o * self.mask_soft + n * new_soft
        self.miss = 0
        return self

    def fade_alpha(self, fade_in_ms):
        age_ms = (time.perf_counter() - self.born) * 1000.0
        return min(1.0, age_ms / max(1.0, fade_in_ms))


class Tracker:
    def __init__(self, iou_thr=0.30, max_miss=30, ema_old=0.75, fade_in_ms=150):
        self.iou_thr = iou_thr
        self.max_miss = max_miss
        self.ema_old = ema_old
        self.fade_in_ms = fade_in_ms
        self.tracks = []

    def update(self, dets):
        """dets: list of (cls_id, conf, box[, mask]). Returns alive tracks."""
        unmatched = set(range(len(self.tracks)))
        for det in dets:
            cls_id, conf, box = det[0], det[1], det[2]
            mask = det[3] if len(det) > 3 else None
            best, best_iou = -1, self.iou_thr
            for ti in unmatched:
                v = iou(self.tracks[ti].box, box)
                if v > best_iou:
                    best, best_iou = ti, v
            if best >= 0:
                self.tracks[best].update(conf, box, self.ema_old, mask)
                self.tracks[best].cls_id = cls_id
                unmatched.discard(best)
            else:
                self.tracks.append(Track(cls_id, conf, box, mask))
        for ti in sorted(unmatched, reverse=True):
            self.tracks[ti].predict()
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return self.tracks

    def predict_only(self):
        for t in self.tracks:
            t.predict()
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return self.tracks
