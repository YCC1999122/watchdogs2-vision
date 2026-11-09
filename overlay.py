"""L3 semantic markers + L4 network links.
game profile: dark navy world / red = hackable / white = info labels."""
import os

import cv2
import numpy as np

from detect import category
from style import NEON_EDGE, NEON_EDGE_HI, STEEL, STEEL_HI

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# COCO class id -> Chinese name
ZH = {0: "人", 1: "自行车", 2: "汽车", 3: "摩托车", 4: "飞机", 5: "公交车",
      6: "火车", 7: "卡车", 8: "船", 9: "红绿灯", 10: "消防栓", 11: "停车标志",
      12: "停车计时器", 13: "长椅", 14: "鸟", 15: "猫", 16: "狗", 17: "马",
      18: "羊", 19: "牛", 20: "大象", 21: "熊", 22: "斑马", 23: "长颈鹿",
      24: "背包", 25: "雨伞", 26: "手提包", 27: "领带", 28: "行李箱", 29: "飞盘",
      30: "滑雪板", 31: "单板", 32: "运动球", 33: "风筝", 34: "棒球棒",
      35: "棒球手套", 36: "滑板", 37: "冲浪板", 38: "网球拍", 39: "瓶子",
      40: "酒杯", 41: "杯子", 42: "叉子", 43: "刀", 44: "勺子", 45: "碗",
      46: "香蕉", 47: "苹果", 48: "三明治", 49: "橙子", 50: "西兰花",
      51: "胡萝卜", 52: "热狗", 53: "披萨", 54: "甜甜圈", 55: "蛋糕",
      56: "椅子", 57: "沙发", 58: "盆栽", 59: "床", 60: "餐桌", 61: "马桶",
      62: "显示器", 63: "笔记本", 64: "鼠标", 65: "遥控器", 66: "键盘",
      67: "手机", 68: "微波炉", 69: "烤箱", 70: "烤面包机", 71: "水槽",
      72: "冰箱", 73: "书", 74: "时钟", 75: "花瓶", 76: "剪刀",
      77: "泰迪熊", 78: "吹风机", 79: "牙刷"}

_font_cache = {}


def _get_font(px=13):
    if px in _font_cache:
        return _font_cache[px]
    f = None
    if _PIL_OK:
        for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                     r"C:\Windows\Fonts\simsun.ttc"):
            if os.path.exists(cand):
                try:
                    f = ImageFont.truetype(cand, px)
                except Exception:
                    f = None
                if f is not None:
                    break
        if f is None:
            f = ImageFont.load_default()
    _font_cache[px] = f
    return f


def _label(img, lines, x, y, color=(232, 238, 242)):
    """small info plate (dark bg + Chinese text) at (x, y), clamped to frame."""
    if not lines or not _PIL_OK:
        return
    font = _get_font(13)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    ws, hs = [], []
    for s in lines:
        b = probe.textbbox((0, 0), s, font=font)
        ws.append(b[2] - b[0])
        hs.append(b[3] - b[1] + 2)
    w = max(ws) + 10
    h = sum(hs) + 6
    H, W = img.shape[:2]
    x = max(0, min(int(x), W - w - 1))
    y = max(0, min(int(y), H - h - 1))
    if w < 4 or h < 4:
        return
    roi = img[y:y + h, x:x + w]
    pil = Image.fromarray(roi[:, :, ::-1].copy())
    d = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])
    d.rectangle([0, 0, w - 1, h - 1], fill=(8, 13, 20), outline=rgb)
    yy = 3
    for s, hh in zip(lines, hs):
        d.text((5, yy), s, font=font, fill=rgb)
        yy += hh
    img[y:y + h, x:x + w] = np.asarray(pil)[:, :, ::-1]

# neon palette
N_HOSTILE = (68, 23, 255)     # #FF1744
N_DEVICE = (255, 255, 24)     # #18FFFF
N_VEHICLE = (0, 196, 255)     # #FFC400

# game palette (BGR)
G_RED = (43, 49, 240)       # anomaly / outsider   (#F0312B)
G_RED_HI = (110, 110, 255)  # anomaly edge
G_BLUE = (243, 150, 33)     # compliant / internal (#2196F3)
G_BLUE_HI = (255, 200, 110) # compliant edge
G_WHITE = (232, 238, 242)   # #E8EEF2 info text
G_STEEL = STEEL


def _brackets(img, x1, y1, x2, y2, color, thick=2, ratio=0.16, scale=1.0):
    w, h = x2 - x1, y2 - y1
    L = int(min(w, h) * ratio * scale)
    if L < 2:
        return
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)


def _blend_rect(img, x1, y1, x2, y2, color, alpha):
    h, w = img.shape[:2]
    x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
    y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    plate = np.full_like(roi, color)
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0, plate, alpha, 0)


def _text(img, s, x, y, scale=0.42, color=G_WHITE, thick=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


# ---------------- neon profile (previous style) ----------------

def _semantics_neon(img, tracks, fade_in_ms):
    order = {id(tr): i + 1 for i, tr in enumerate(
        sorted(tracks, key=lambda x: x.id))}
    colors = {"hostile": N_HOSTILE, "device": N_DEVICE, "vehicle": N_VEHICLE,
              "infra": G_STEEL, "other": N_DEVICE}
    for tr in tracks:
        disp = order[id(tr)]
        cat = category(tr.cls_id)
        color = colors[cat]
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        fade = tr.fade_alpha(fade_in_ms)
        scale = 0.3 + 0.7 * fade
        _blend_rect(img, x1, y1, x2, y2, color, 0.08 * fade)
        if cat == "hostile":
            _brackets(img, x1, y1, x2, y2, color, 2, 0.16, scale)
            _text(img, f"TARGET-{disp:02d}", x1, max(16, y1 - 20), 0.45, color)
            _text(img, f"THREAT:MED {tr.conf:.2f}", x1, max(16, y1 - 7),
                  0.40, color)
        elif cat == "vehicle":
            _brackets(img, x1, y1, x2, y2, color, 1, 0.12, scale)
            _text(img, f"VEH-{disp:02d}", x1, max(16, y1 - 7), 0.42, color)
        else:
            _brackets(img, x1, y1, x2, y2, color, 1, 0.14, scale)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(img, (cx - 2, cy - 2), (cx + 2, cy + 2), color, -1,
                          cv2.LINE_AA)
            _text(img, f"NODE-{disp:02X} LINKED", x1, max(16, y1 - 7),
                  0.42, color)
    return img


def _dash_line(img, p, q, dash, gap, phase, color, thick):
    dist = np.hypot(q[0] - p[0], q[1] - p[1])
    if dist < 2:
        return
    unit = dash + gap
    k = int(phase) % unit
    while k < dist:
        t0 = k / dist
        t1 = min(1.0, (k + dash) / dist)
        a = (int(p[0] + (q[0] - p[0]) * t0), int(p[1] + (q[1] - p[1]) * t0))
        b = (int(p[0] + (q[0] - p[0]) * t1), int(p[1] + (q[1] - p[1]) * t1))
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)
        k += unit


def _network_neon(img, tracks, t, prm):
    h, w = img.shape[:2]
    nodes = [(int(c[0]), int(c[1])) for c in (tr.center() for tr in tracks)]
    if not nodes:
        return img
    anchors = [(int(w * .05), int(h * .06)), (int(w * .95), int(h * .06)),
               (int(w * .05), int(h * .94)), (int(w * .95), int(h * .94))]
    link_k = prm.get("node_link_k", 2)
    pairs = set()
    for i, n in enumerate(nodes):
        d = sorted([(np.hypot(n[0] - m[0], n[1] - m[1]), j)
                    for j, m in enumerate(nodes) if j != i])[:link_k]
        for _, j in d:
            pairs.add((min(i, j), max(i, j)))
        a = min(range(4), key=lambda k: np.hypot(n[0] - anchors[k][0],
                                                 n[1] - anchors[k][1]))
        pairs.add((i, -(a + 1)))
    ov = np.zeros_like(img)
    for i, j in pairs:
        p = nodes[i]
        q = anchors[-j - 1] if j < 0 else nodes[j]
        cv2.line(ov, p, q, NEON_EDGE, 3, cv2.LINE_AA)
    img[:] = cv2.addWeighted(img, 1.0, ov, 0.12, 0)
    phase = t * prm.get("dash_speed", 40)
    for i, j in pairs:
        p = nodes[i]
        q = anchors[-j - 1] if j < 0 else nodes[j]
        _dash_line(img, p, q, prm.get("dash_px", 6), prm.get("dash_gap", 6),
                   phase, NEON_EDGE_HI, 1)
    for i, n in enumerate(nodes):
        glow = 0.6 + 0.4 * np.sin(t * 2 + i * 1.7)
        cv2.circle(img, n, 3, NEON_EDGE_HI, -1, cv2.LINE_AA)
        c = tuple(int(v * glow) for v in N_DEVICE)
        cv2.circle(img, n, 7, c, 1, cv2.LINE_AA)
    return img


# ---------------- game profile (layer view: silhouette recolor) ----------------

def cat_of(tr):
    """category after fine-classifier correction (run.py sets cat_override)."""
    return getattr(tr, "cat_override", None) or category(tr.cls_id)


def obj_name(tr):
    """best available display name: fine classifier > COCO zh table."""
    return getattr(tr, "fine_zh", None) or ZH.get(tr.cls_id, "物品")


def _status_colors(tr, prm):
    """library override first (face match / device whitelist set by run.py),
    then category defaults, then the demo policy for people & animals."""
    ov = getattr(tr, "status_override", None)
    if ov == "ok":
        return (G_BLUE, G_BLUE_HI)
    if ov == "bad":
        return (G_RED, G_RED_HI)
    cat = cat_of(tr)
    if cat in ("device", "infra"):
        return (G_BLUE, G_BLUE_HI)
    policy = prm.get("demo_policy", "alternate")
    if policy == "all_red":
        bad = True
    elif policy == "all_blue":
        bad = False
    else:
        bad = (tr.id % 3 == 2)
    return (G_RED, G_RED_HI) if bad else (G_BLUE, G_BLUE_HI)


def _silhouette_mask(tr, shape):
    """temporally-smoothed soft mask (float 0..1), or geometric fallback."""
    h, w = shape[:2]
    m = getattr(tr, "mask_soft", None)
    if m is not None and m.shape == (h, w) and m.max() > 0.05:
        return m
    x1, y1, x2, y2 = [int(v) for v in tr.box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    m = np.zeros((h, w), np.float32)
    if category(tr.cls_id) in ("hostile", "animal"):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.ellipse(m, (cx, cy), (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)),
                    0, 0, 360, 1.0, -1)
    else:
        cv2.rectangle(m, (x1, y1), (x2, y2), 1.0, -1)
    return m


def _fill_silhouette(img, soft, color, hi_color, fade=1.0,
                     a_top=0.60, a_bot=0.38):
    """recolor the object with the temporally-smoothed soft mask as weight,
    so the silhouette edges fade smoothly instead of flickering frame to frame."""
    soft = np.clip(soft, 0.0, 1.0)
    ys, xs = np.nonzero(soft > 0.02)
    if not len(ys):
        return
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    src = img[y1:y2, x1:x2].astype(np.float32)
    m = soft[y1:y2, x1:x2]
    rh = y2 - y1
    grad = (np.linspace(a_top, a_bot, rh, dtype=np.float32) * fade)[:, None, None]
    lum = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tex = (0.50 + 0.50 * lum)[..., None]
    fill = np.array(color, np.float32) * tex
    # soft mask drives the fill weight everywhere -> no hard boundary, no flicker
    a = grad * m[..., None]
    out = src * (1 - a) + fill * a
    # bright edge along mid-level contour of the soft mask
    edge = cv2.morphologyEx((m > 0.4).astype(np.uint8) * 255,
                            cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ea = 0.85 * fade
    out[edge] = out[edge] * (1 - ea) + np.array(hi_color, np.float32) * ea
    img[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)


def _person_info(tr, frame_h):
    """rough estimate from box geometry + velocity (not face recognition):
    body size (adult/child by pixel height), posture, motion."""
    x1, y1, x2, y2 = tr.box
    bw, bh = x2 - x1, y2 - y1
    hr = bh / max(1, frame_h)
    size = "成人" if hr > 0.16 else "儿童/远景"
    form = "站立" if bh > bw * 1.5 else "蹲坐"
    vx, vy = tr.vel
    move = "行走" if (vx * vx + vy * vy) ** 0.5 > 2.5 else "静止"
    return size, form, move


def _semantics_game(img, tracks, prm):
    fade_in_ms = prm.get("fade_in_ms", 150)
    label_on = prm.get("label_on", True)
    fh = img.shape[0]
    for tr in tracks:
        cat = cat_of(tr)
        fade = tr.fade_alpha(fade_in_ms)
        name = obj_name(tr)
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        if cat in ("hostile", "animal", "device", "infra"):
            mask = _silhouette_mask(tr, img.shape)
            if mask is None:
                continue
            color, hi = _status_colors(tr, prm)
            _fill_silhouette(img, mask, color, hi, fade)
            if label_on:
                bad = (hi == G_RED_HI)
                status = "异常" if bad else "合规"
                if cat == "hostile":
                    size, form, move = _person_info(tr, fh)
                    fname = getattr(tr, "face_name", None)
                    frole = getattr(tr, "face_role", None)
                    who = f"{fname}·{frole}" if fname else "未知人员"
                    _label(img, [f"{who} #{tr.id} {size}", f"{form}·{move}·{status}"],
                           x1, max(2, y1 - 40), hi)
                elif cat == "animal":
                    _label(img, [f"{name} #{tr.id}", status],
                           x1, max(2, y1 - 40), hi)
                else:
                    _label(img, [f"{name} #{tr.id}", f"电子设备·{status}·已填色"],
                           x1, max(2, y1 - 40), hi)
        elif cat == "vehicle":
            _brackets(img, x1, y1, x2, y2, G_STEEL, 1, 0.10, 0.5 + 0.5 * fade)
            if label_on:
                _label(img, [f"{name}", "未填色"], x1, max(2, y1 - 40), G_STEEL)
        elif label_on:
            _label(img, [f"{name}", "背景物·未填色"], x1, max(2, y1 - 40),
                   (140, 150, 158))
    return img


def _network_game(img, tracks, t, prm):
    """dashed links between electronic/device nodes, colored by status."""
    dev = [tr for tr in tracks if cat_of(tr) in ("device", "infra")]
    nodes = [(int(c[0]), int(c[1])) for c in (tr.center() for tr in dev)]
    if not nodes:
        return img
    h, w = img.shape[:2]
    anchors = [(w // 2, int(h * .04))]     # single top uplink
    pairs = set()
    for i, n in enumerate(nodes):
        d = sorted([(np.hypot(n[0] - m[0], n[1] - m[1]), j)
                    for j, m in enumerate(nodes) if j != i])[:1]
        for _, j in d:
            pairs.add((min(i, j), max(i, j)))
        if len(nodes) == 1:
            pairs.add((i, -1))
    phase = t * prm.get("dash_speed", 30)
    for i, j in pairs:
        p = nodes[i]
        q = anchors[-j - 1] if j < 0 else nodes[j]
        color = _status_colors(dev[i], prm)[1]
        _dash_line(img, p, q, prm.get("dash_px", 4), prm.get("dash_gap", 6),
                   phase, color, 1)
    for i, n in enumerate(nodes):
        color = _status_colors(dev[i], prm)[1]
        cv2.rectangle(img, (n[0] - 3, n[1] - 3), (n[0] + 3, n[1] + 3),
                      color, -1, cv2.LINE_AA)
    return img


# ---------------- dispatch ----------------

def draw_semantics(img, tracks, t, prm):
    if prm.get("profile") == "game":
        return _semantics_game(img, tracks, prm)
    return _semantics_neon(img, tracks, prm.get("fade_in_ms", 150))


def draw_network(img, tracks, t, prm):
    if prm.get("profile") == "game":
        return _network_game(img, tracks, t, prm)
    return _network_neon(img, tracks, t, prm)
