"""NetHack vision: layer-view conversion pipeline.

Usage:
  python run.py --source input.mp4 --out output.mp4 --compare
  python run.py --source 0 --display                       # camera
  python app.py                                             # GUI
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import detect
import hud
import overlay
import style
from track import Tracker

try:
    import library as libmod
    from face import FaceEngine
    _FACE_OK = True
except Exception:
    _FACE_OK = False

try:
    from labeler import CLOTHING, ELECTRONIC, PERSONISH, FineLabeler, display_name
    _LABEL_OK = True
except Exception:
    _LABEL_OK = False

OUT_MAX = 1280          # output longest side; short side scaled to keep aspect


def load_config(path, profile_override=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = profile_override or cfg.get("profile", "game")
    prm = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    prm.update(cfg.get(name, {}))
    prm["profile"] = name
    return prm


def _fit_size(sw, sh, max_side):
    """scale (sw,sh) so longest side == max_side, keep aspect."""
    if sw >= sh:
        w = max_side
        h = int(round(sh * max_side / sw))
    else:
        h = max_side
        w = int(round(sw * max_side / sh))
    return w, h


def open_source(source):
    if isinstance(source, str) and source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        for _ in range(10):
            cap.read()
        return cap, None, 640, 480, 30.0
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 30.0
    return cap, total, sw, sh, float(fps)


def label_panel(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (10, 16, 20), -1)
    cv2.putText(img, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (232, 238, 242), 1, cv2.LINE_AA)
    return img


def _cat(tr):
    from detect import category
    return getattr(tr, "cat_override", None) or category(tr.cls_id)


def _refine_labels(objs, frame, frame_no, labeler):
    """run the 1000-class classifier on each track's crop (once per track,
    refreshed periodically). Fixes person false-positives and names objects
    COCO does not know (teacup, dish antenna, ...)."""
    if labeler is None:
        return
    fh, fw = frame.shape[:2]
    for tr in objs:
        last = getattr(tr, "fine_frame", -10 ** 9)
        done = getattr(tr, "fine_done", False)
        if done and frame_no - last < 45:
            continue
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        mx = int((x2 - x1) * 0.12)
        my = int((y2 - y1) * 0.12)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(fw - 1, x2 + mx), min(fh - 1, y2 + my)
        if x2 - x1 < 12 or y2 - y1 < 12:
            continue
        crop = frame[y1:y2, x1:x2]
        try:
            en, conf = labeler.classify(crop)
        except Exception:
            continue
        tr.fine_done = True
        tr.fine_frame = frame_no
        if conf < 0.18:                    # too unsure -> keep YOLO's label
            continue
        tr.fine_en = en
        tr.fine_conf = conf
        tr.fine_zh = display_name(en)
        from detect import category
        ycat = category(tr.cls_id)
        # --- fix categories using the finer label ---
        if en in ELECTRONIC and ycat in ("other", "animal", "infra"):
            tr.cat_override = "device"
        elif en in PERSONISH and ycat != "hostile" and conf > 0.45:
            tr.cat_override = "hostile"
        elif (ycat == "hostile" and en not in PERSONISH
              and en not in CLOTHING and en not in ELECTRONIC and conf > 0.55):
            # YOLO confidently mistook an object for a person -> demote.
            # (clothing/electronics often classify on real people -> protected)
            tr.cat_override = "other"


def _resolve_status(objs, frame, frame_no, face_eng, faces_lib, dev_allow):
    """library-driven status: face match for persons, whitelist for devices,
    anomaly logging for unknown targets. Sets tr.status_override / tr.face_name."""
    fh, fw = frame.shape[:2]
    for tr in objs:
        cat = _cat(tr)
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
        if cat == "hostile":
            if face_eng is not None:
                tried = getattr(tr, "face_tries", 0)
                if tried < 3 and frame_no % 10 == (tr.id % 10) and x2 - x1 > 24:
                    tr.face_tries = tried + 1
                    crop = frame[y1:y2, x1:x2]
                    name, role, score, aligned = face_eng.identify(crop, faces_lib)
                    if name:
                        tr.face_name, tr.face_role = name, role
                        tr.status_override = "ok" if role == "内部" else "bad"
                    elif aligned is not None:
                        tr.status_override = "bad"
                        if tr.face_tries >= 2 and not getattr(tr, "anom_logged", False):
                            tr.anom_logged = True
                            libmod.log_anomaly("face", f"未知人员 #{tr.id}", aligned)
            elif not hasattr(tr, "status_override"):
                tr.status_override = None
        elif cat == "device":
            ok = tr.cls_id in dev_allow
            tr.status_override = "ok" if ok else "bad"
            if not ok and not getattr(tr, "anom_logged", False) and x2 - x1 > 8:
                tr.anom_logged = True
                crop = frame[max(0, y1 - 6):min(fh, y2 + 6),
                             max(0, x1 - 6):min(fw, x2 + 6)]
                if crop.size:
                    libmod.log_anomaly(
                        "device",
                        f"未知设备 {overlay.ZH.get(tr.cls_id, tr.cls_id)}",
                        crop, extra={"cls": int(tr.cls_id)})
        elif cat == "infra":
            tr.status_override = "ok"


def process_video(source, out=None, compare=False, profile=None,
                  config=None, model=None, display=False, max_frames=0,
                  detect_every=None, progress=None, log=print):
    """Run the full pipeline on one source.

    progress(frame_no, total) -> called per frame (may be None).
    log(msg) -> stdout replacement for GUI embedding.
    Returns number of frames processed.
    """
    config = config or os.path.join(HERE, "config.yaml")
    prm = load_config(config, profile)
    pw, ph = prm["proc_width"], prm["proc_height"]
    de = detect_every if detect_every is not None else prm["detect_every"]

    detector = detect.Detector(model or detect.default_model(HERE),
                               prm["detect_input"],
                               prm["detect_conf"], prm["nms_iou"])
    tracker = Tracker(prm["track_iou"], prm["track_max_miss"],
                      prm["ema_old"], prm["fade_in_ms"])

    # face recognition + device whitelist + anomaly library
    face_eng, faces_lib, dev_allow = None, [], set()
    if _FACE_OK:
        try:
            lib = libmod.load()
            faces_lib = lib["faces"]
            dev_allow = set(lib["devices"]["allowed"])
            face_eng = FaceEngine(
                os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx"),
                os.path.join(HERE, "models", "face_recognition_sface_2021dec.onnx"))
            if not face_eng.ok:
                face_eng = None
        except Exception:
            face_eng = None

    # wide-vocabulary fine classifier (1000 ImageNet classes)
    labeler = None
    if _LABEL_OK:
        try:
            labeler = FineLabeler(
                os.path.join(HERE, "models", "mobilenetv2-7.onnx"),
                os.path.join(HERE, "models", "imagenet_classes.txt"))
            if not labeler.ok:
                labeler = None
        except Exception:
            labeler = None

    cap, total, sw, sh, src_fps = open_source(source)
    if not cap.isOpened():
        log(f"cannot open source: {source}")
        return 0
    ow, oh = _fit_size(sw, sh, OUT_MAX)
    log(f"source {sw}x{sh} @ {src_fps:.2f}fps  ->  out {ow}x{oh}  "
        f"profile={prm['profile']}")

    writer = None
    if out:
        vw = ow * 2 if compare else ow
        # match source fps so output duration == input duration
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (vw, oh))

    if display:
        cv2.namedWindow("NetHack", cv2.WINDOW_NORMAL)

    t0 = time.perf_counter()
    stage_ms = {k: [] for k in ("detect", "style", "overlay", "hud", "total")}
    frame_no = 0
    fps_ema = 0.0

    try:
        while True:
            f0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (pw, ph))
            t = time.perf_counter() - t0

            if frame_no % de == 0:
                objs = tracker.update(detector(frame))
            else:
                objs = tracker.predict_only()
            _refine_labels(objs, frame, frame_no, labeler)
            _resolve_status(objs, frame, frame_no, face_eng, faces_lib, dev_allow)
            t_detect = time.perf_counter()

            out_img = style.base_grade(frame, prm)
            out_img = style.edges_overlay(out_img, frame, prm)
            t_style = time.perf_counter()

            out_img = overlay.draw_semantics(out_img, objs, t, prm)
            out_img = overlay.draw_network(out_img, objs, t, prm)
            t_overlay = time.perf_counter()

            out_img = hud.finish(out_img, len(objs), t, prm, fps_ema, frame_no, "ar")
            t_hud = time.perf_counter()

            out_img = cv2.resize(out_img, (ow, oh))
            if compare:
                orig = cv2.resize(frame, (ow, oh))
                view = np.hstack([label_panel(orig.copy(), "ORIGINAL"),
                                  label_panel(out_img, f"LAYER VIEW [{prm['profile']}]")])
            else:
                view = out_img

            if display:
                cv2.imshow("NetHack", view)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if writer:
                writer.write(view)
            t_write = time.perf_counter()

            ms = (t_write - f0) * 1000.0
            stage_ms["detect"].append((t_detect - f0) * 1000.0)
            stage_ms["style"].append((t_style - t_detect) * 1000.0)
            stage_ms["overlay"].append((t_overlay - t_style) * 1000.0)
            stage_ms["hud"].append((t_hud - t_overlay) * 1000.0)
            stage_ms["total"].append(ms)
            fps_ema = 1000.0 / np.mean(stage_ms["total"][-30:])

            frame_no += 1
            if progress:
                progress(frame_no, total or frame_no)
            if max_frames and frame_no >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    if not stage_ms["total"]:
        log("no frames processed")
        return 0
    log(f"frames: {frame_no}  avg FPS: {1000.0 / np.mean(stage_ms['total']):.1f}")
    return frame_no


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--profile", default=None, choices=["game", "neon"])
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--detect-every", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    process_video(args.source, out=args.out, compare=args.compare,
                  profile=args.profile, config=args.config, model=args.model,
                  display=args.display, max_frames=args.max_frames,
                  detect_every=args.detect_every)


if __name__ == "__main__":
    main()
