"""YuNet face detection + SFace embedding + library matching (OpenCV builtin)."""
import os

import cv2
import numpy as np


class FaceEngine:
    def __init__(self, det_path, rec_path, score_thr=0.8, match_thr=0.40):
        self.ok = os.path.exists(det_path) and os.path.exists(rec_path)
        if not self.ok:
            return
        self.det = cv2.FaceDetectorYN_create(det_path, "", (320, 320),
                                             score_thr, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF_create(rec_path, "")
        self.match_thr = match_thr

    def find_face(self, bgr):
        """largest face -> aligned 112x112 crop, or None."""
        h, w = bgr.shape[:2]
        if w < 20 or h < 20:
            return None
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        return self.rec.alignCrop(bgr, face)

    def embed(self, aligned):
        f = self.rec.feature(aligned)[0].astype(np.float32)
        n = float(np.linalg.norm(f))
        return f / n if n > 0 else f

    def identify(self, bgr, library):
        """-> (name, role, score, aligned_crop). name=None if unknown."""
        al = self.find_face(bgr)
        if al is None:
            return None, None, 0.0, None
        emb = self.embed(al)
        best, bs = None, -1.0
        for entry in library:
            e = np.asarray(entry["embedding"], np.float32)
            n = float(np.linalg.norm(e))
            if n == 0:
                continue
            s = float(np.dot(emb, e / n))
            if s > bs:
                bs, best = s, entry
        if best is not None and bs >= self.match_thr:
            return best["name"], best.get("role", "内部"), bs, al
        return None, None, bs, al
