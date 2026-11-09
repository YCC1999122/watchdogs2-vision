"""L5 HUD + post effects.
game profile: film grain + vignette + minimal white text only.
neon profile: scanlines / grid / crosshair / chroma / band."""
import cv2
import numpy as np

C_EDGE_HI = (240, 233, 127)
C_DEVICE = (255, 214, 47)
C_TEXT = (250, 247, 230)
G_WHITE = (232, 238, 242)

_noise_cache = {}
_vignette_cache = {}


def _noise(h, w, sigma, slot):
    key = (h, w, sigma)
    if key not in _noise_cache:
        rng = np.random.default_rng(7)
        _noise_cache[key] = [rng.normal(0, sigma, (h, w, 3)).astype(np.float32)
                             for _ in range(3)]
    return _noise_cache[key][slot % 3]


def _vignette(h, w, strength):
    key = (h, w, strength)
    if key not in _vignette_cache:
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        v = np.clip(1.0 - strength * np.clip(r - 0.4, 0, 2) ** 1.5, 0, 1)
        _vignette_cache[key] = v[..., None].astype(np.float32)
    return _vignette_cache[key]


def _finish_game(img, n_nodes, t, fps, frame_no, prm):
    h, w = img.shape[:2]
    # film grain
    sigma = prm.get("noise_sigma", 5)
    if sigma > 0:
        img = np.clip(img.astype(np.float32) + _noise(h, w, sigma, frame_no),
                      0, 255).astype(np.uint8)
    # vignette
    img = (img.astype(np.float32)
           * _vignette(h, w, prm.get("vignette", 0.30))).astype(np.uint8)
    # minimal white text
    cv2.putText(img, "LAYER VIEW // AREA MONITOR", (18, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, G_WHITE, 1, cv2.LINE_AA)
    tc = f"{int(t // 60):02d}:{int(t % 60):02d}:{int(t % 1 * 30):02d}"
    for i, s in enumerate([f"NODES {n_nodes:02d}", f"FPS {fps:4.1f}", tc]):
        cv2.putText(img, s, (w - 120, h - 46 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, G_WHITE, 1, cv2.LINE_AA)
    return img


def _finish_neon(img, n_nodes, t, fps, frame_no, prm, mode):
    h, w = img.shape[:2]
    if mode == "wire":
        img = cv2.addWeighted(img, 0.25, np.zeros_like(img), 0.75, 0)
    grid_step = prm.get("grid_step", 60)
    ov = img.copy()
    for x in range(0, w, grid_step):
        cv2.line(ov, (x, 0), (x, h), (70, 90, 90), 1)
    for y in range(0, h, grid_step):
        cv2.line(ov, (0, y), (w, y), (70, 90, 90), 1)
    img = cv2.addWeighted(ov, 0.06, img, 0.94, 0)
    L = int(min(w, h) * 0.08)
    for cx, cy, dx, dy in [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), C_EDGE_HI, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), C_EDGE_HI, 2, cv2.LINE_AA)
    cx, cy = w // 2, h // 2
    for x0, y0, x1, y1 in [(-26, 0, -10, 0), (26, 0, 10, 0), (0, -26, 0, -10), (0, 26, 0, 10)]:
        cv2.line(img, (cx + x0, cy + y0), (cx + x1, cy + y1), C_TEXT, 1, cv2.LINE_AA)
    img[::3] = (img[::3].astype(np.float32)
                * (1.0 - prm.get("scanline", 0.30))).astype(np.uint8)
    band_period = prm.get("scan_band_period", 8.0)
    band_h = 60
    span = h + band_h * 2
    band = int(((t % band_period) / band_period) * span) - band_h
    ov2 = img.copy()
    cv2.rectangle(ov2, (0, band), (w, min(h, band + band_h)), C_EDGE_HI, -1)
    img = cv2.addWeighted(ov2, 0.08, img, 0.92, 0)
    img = (img.astype(np.float32)
           * _vignette(h, w, prm.get("vignette", 0.35))).astype(np.uint8)
    px = prm.get("chroma_px", 2)
    if px > 0:
        b, g, r = cv2.split(img)
        img = cv2.merge([np.roll(b, px, axis=1), g, np.roll(r, -px, axis=1)])
    sigma = prm.get("noise_sigma", 4)
    if sigma > 0:
        img = np.clip(img.astype(np.float32) + _noise(h, w, sigma, frame_no),
                      0, 255).astype(np.uint8)
    cv2.putText(img, "NETHACK // ONLINE", (24, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, C_EDGE_HI, 1, cv2.LINE_AA)
    cv2.putText(img, "SCAN ACTIVE", (24, 56),
                cv2.FONT_HERSHEY_PLAIN, 0.9, C_DEVICE, 1, cv2.LINE_AA)
    tc = f"TC {int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}:{int(t % 1 * 30):02d}"
    lines = [f"NODES {n_nodes:02d}", "BOTNET 12/20", f"FPS {fps:4.1f}", tc]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (w - 175, h - 74 + i * 18),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, C_TEXT, 1, cv2.LINE_AA)
    return img


def finish(img, n_nodes, t, prm, fps=0.0, frame_no=0, mode="ar"):
    if prm.get("profile") == "game":
        return _finish_game(img, n_nodes, t, fps, frame_no, prm)
    return _finish_neon(img, n_nodes, t, fps, frame_no, prm, mode)
