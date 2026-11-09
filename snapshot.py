"""Render still frames of the NetHack view (converted side only) for review.

Usage:
  python snapshot.py --source test_assets/raw/street_city_855564.mp4 --frame 230 --out output/nethack_frame.png
"""
import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import detect
import hud
import overlay
import style
from run import load_config
from track import Tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=230)
    ap.add_argument("--out", default=os.path.join(HERE, "output", "nethack_frame.png"))
    ap.add_argument("--profile", default=None, choices=["game", "neon"])
    ap.add_argument("--mode", default="ar", choices=["ar", "wire"])
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--model", default=None, help="onnx path (default: auto, seg preferred)")
    args = ap.parse_args()

    prm = load_config(args.config, args.profile)
    pw, ph = prm["proc_width"], prm["proc_height"]
    ow, oh = prm["out_width"], prm["out_height"]

    det = detect.Detector(args.model or detect.default_model(HERE),
                          prm["detect_input"],
                          prm["detect_conf"], prm["nms_iou"])
    trk = Tracker(prm["track_iou"], prm["track_max_miss"],
                  prm["ema_old"], prm["fade_in_ms"])

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    objs, frame = [], None
    for i in range(args.frame + 1):
        ok, fr = cap.read()
        if not ok:
            break
        frame = cv2.resize(fr, (pw, ph))
        if i % prm["detect_every"] == 0:
            objs = trk.update(det(frame))
        else:
            objs = trk.predict_only()
    cap.release()
    if frame is None:
        raise SystemExit("frame read failed")

    t = args.frame / fps
    out = style.base_grade(frame, prm)
    out = style.edges_overlay(out, frame, prm)
    out = overlay.draw_semantics(out, objs, t, prm)
    out = overlay.draw_network(out, objs, t, prm)
    out = hud.finish(out, len(objs), t, prm, 24.0, args.frame, args.mode)
    out = cv2.resize(out, (ow, oh))
    cv2.imwrite(args.out, out)
    print(f"saved {args.out}  profile={prm['profile']}  tracks={len(objs)}")


if __name__ == "__main__":
    main()
