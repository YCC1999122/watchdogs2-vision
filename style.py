"""L1 base grading + L2 structure edges.
Profiles: "game" (Watch Dogs 2 NetHack authentic) / "neon" (fluorescent)."""
import cv2
import numpy as np

# neon palette (BGR)
NEON_EDGE = (255, 229, 0)       # #00E5FF
NEON_EDGE_HI = (255, 255, 179)  # #B3FFFF

# game palette (BGR) — dark navy world, steel-blue lines
NAVY_LO = (26, 15, 7)       # #070F1A
NAVY_HI = (66, 44, 24)      # #182C42
STEEL = (122, 92, 62)       # #3E5C7A
STEEL_HI = (152, 122, 92)   # #5C7A98


def base_grade(frame, prm):
    """L1 base grading."""
    if prm.get("profile") == "game":
        # dark navy digital world: heavy desaturation, strong crush
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g = np.clip(g, 0, prm.get("highlight_clip", 0.90)) ** prm.get("gamma", 1.30)
        lo = np.array(NAVY_LO, np.float32)
        hi = np.array(NAVY_HI, np.float32)
        return np.clip(lo + (hi - lo) * g[..., None], 0, 255).astype(np.uint8)
    # neon profile
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = np.power(np.clip(g, 0, prm.get("highlight_clip", 0.85)),
                 prm.get("gamma", 0.95))
    tint = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    gray3 = cv2.merge([(g * 255).astype(np.uint8)] * 3)
    a = prm.get("base_alpha", 0.75)
    base = cv2.addWeighted(tint, a, gray3, 1.0 - a, 0)
    b, gg, r = cv2.split(base.astype(np.float32))
    out = cv2.merge([np.clip(b + prm.get("shift_b", 12), 0, 255),
                     np.clip(gg + prm.get("shift_g", 4), 0, 255),
                     np.clip(r + prm.get("shift_r", -8), 0, 255)])
    return np.clip(out * prm.get("base_darken", 0.85), 0, 255).astype(np.uint8)


def edges_overlay(base, frame, prm):
    """L2 structure lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    if prm.get("profile") == "game":
        # thin steel-blue lines, no glow, no dilation
        e1 = cv2.Canny(blur, prm.get("canny_lo", 60), prm.get("canny_hi", 140))
        e2 = cv2.Canny(blur, prm.get("canny_hi2_lo", 150), prm.get("canny_hi2_hi", 220))
        out = base.astype(np.float32)
        a1, a2 = prm.get("edge_alpha", 0.45), prm.get("edge_hi_alpha", 0.70)
        m1 = e1 > 0
        out[m1] = out[m1] * (1 - a1) + np.array(STEEL, np.float32) * a1
        m2 = e2 > 0
        out[m2] = out[m2] * (1 - a2) + np.array(STEEL_HI, np.float32) * a2
        return np.clip(out, 0, 255).astype(np.uint8)
    # neon profile
    e1 = cv2.dilate(cv2.Canny(blur, prm.get("canny_lo", 40), prm.get("canny_hi", 110)),
                    np.ones((2, 2), np.uint8))
    e2 = cv2.Canny(blur, prm.get("canny_hi2_lo", 120), prm.get("canny_hi2_hi", 200))
    out = base.astype(np.float32)
    glow_alpha = prm.get("edge_glow", 0.18)
    if glow_alpha > 0:
        glow = cv2.GaussianBlur(e1, (5, 5), 0).astype(np.float32) / 255.0
        out = out * (1 - glow[..., None] * glow_alpha) + \
            np.array(NEON_EDGE, np.float32) * (glow[..., None] * glow_alpha)
    a1, a2 = prm.get("edge_alpha", 0.50), prm.get("edge_hi_alpha", 0.90)
    m1 = e1 > 0
    out[m1] = out[m1] * (1 - a1) + np.array(NEON_EDGE, np.float32) * a1
    m2 = e2 > 0
    out[m2] = out[m2] * (1 - a2) + np.array(NEON_EDGE_HI, np.float32) * a2
    return np.clip(out, 0, 255).astype(np.uint8)
