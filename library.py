"""Persistent libraries: known faces, allowed device types, anomaly log.

Stored in <app>/library/ (next to the exe when frozen).
  faces.json      [{"name","role","embedding":[...],"image"}]
  devices.json    {"allowed": [cls_id,...]}
  anomalies.jsonl one JSON per line {"ts","kind","label","image","status"}
  images/         face photos & anomaly crops
"""
import json
import os
import sys
import time

if getattr(sys, "frozen", False):
    BASE = os.path.join(os.path.dirname(sys.executable), "library")
else:
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

FACES_JSON = os.path.join(BASE, "faces.json")
DEVICES_JSON = os.path.join(BASE, "devices.json")
ANOM_JSONL = os.path.join(BASE, "anomalies.jsonl")

# device classes that are allowed by default (common company devices)
DEFAULT_ALLOWED = [62, 63, 64, 65, 66, 67, 72]   # tv laptop mouse remote
                                                   # keyboard phone refrigerator


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load():
    faces = _load_json(FACES_JSON, [])
    devices = _load_json(DEVICES_JSON, {"allowed": DEFAULT_ALLOWED})
    return {"faces": faces, "devices": devices}


def add_face(name, role, embedding, image_bgr):
    import cv2
    faces = _load_json(FACES_JSON, [])
    fid = int(time.time() * 1000)
    img_name = f"face_{fid}.jpg"
    cv2.imwrite(os.path.join(IMG_DIR, img_name), image_bgr)
    faces.append({"id": fid, "name": name, "role": role,
                  "embedding": [float(v) for v in embedding], "image": img_name})
    _save_json(FACES_JSON, faces)
    return fid


def delete_face(fid):
    faces = _load_json(FACES_JSON, [])
    faces = [f for f in faces if f.get("id") != fid]
    _save_json(FACES_JSON, faces)


def set_allowed_devices(cls_ids):
    _save_json(DEVICES_JSON, {"allowed": [int(c) for c in cls_ids]})


def allowed_devices():
    return set(_load_json(DEVICES_JSON, {"allowed": DEFAULT_ALLOWED})["allowed"])


def log_anomaly(kind, label, image_bgr, extra=None):
    """record an unknown face/device; returns anomaly id."""
    import cv2
    aid = int(time.time() * 1000)
    img_name = f"anom_{aid}.jpg"
    cv2.imwrite(os.path.join(IMG_DIR, img_name), image_bgr)
    entry = {"id": aid, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
             "kind": kind, "label": label, "image": img_name, "status": "pending"}
    if extra:
        entry.update(extra)
    with open(ANOM_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return aid


def list_anomalies():
    if not os.path.exists(ANOM_JSONL):
        return []
    out = []
    with open(ANOM_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out[::-1]     # newest first


def _rewrite_anomalies(entries):
    with open(ANOM_JSONL, "w", encoding="utf-8") as f:
        for e in reversed(entries):     # restore chronological order
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def set_anomaly_status(aid, status):
    entries = list_anomalies()
    for e in entries:
        if e.get("id") == aid:
            e["status"] = status
    _rewrite_anomalies(entries)


def get_anomaly_image(aid):
    for e in list_anomalies():
        if e.get("id") == aid:
            return os.path.join(IMG_DIR, e["image"])
    return None
