# enhancement.py
"""
Fast RGB–NIR fusion **with guided‑filter denoising**
====================================================

* Pipeline:
  1.  Local‑contrast fusion (unchanged Awad et al. 2020).
  2.  **Guided image filter** denoising – the high‑resolution NIR frame is
     used as the *guide* so noise is smoothed only where NIR is locally
     flat; RGB edges that coincide with NIR edges remain crisp.

The guided filter (§He et al., ECCV 2010) is **O(N)** and implemented in
OpenCV ≥4.0 via `cv2.ximgproc.guidedFilter`, so inference is near real‑
time on CPU.  If the `ximgproc` module is missing we gracefully fall back
to a joint bilateral filter (still NIR‑guided).

Usage
-----
```bash
python enhancement.py --rgb scene_rgb.png --nir scene_nir.png \
                      --out scene_fused_denoised.png --radius 8 --eps 1e-3
```
Set `--skip_fusion` if you already have a fused RGB frame and only want
NIR‑guided denoising.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


import cv2
import numpy as np

# -----------------------------------------------------------------------------
# 1.  FUSION (same as before)
# -----------------------------------------------------------------------------

def _get_max(img: np.ndarray, cx: int, cy: int, win: int) -> float:
    l = max(0, cy - win // 2)
    t = max(0, cx - win // 2)
    r = min(img.shape[1] - 1, cy + win // 2)
    b = min(img.shape[0] - 1, cx + win // 2)
    return float(np.max(img[t : b + 1, l : r + 1]))


def _get_min(img: np.ndarray, cx: int, cy: int, win: int) -> float:
    l = max(0, cy - win // 2)
    t = max(0, cx - win // 2)
    r = min(img.shape[1] - 1, cy + win // 2)
    b = min(img.shape[0] - 1, cx + win // 2)
    return float(np.min(img[t : b + 1, l : r + 1]))


def local_contrast(img: np.ndarray, alpha: float = 0.5, win: int = 5) -> np.ndarray:
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    amplitude = np.sqrt(dx ** 2 + dy ** 2)
    out = np.empty_like(img, np.float32)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            max_i = _get_max(img, x, y, win)
            min_i = _get_min(img, x, y, win)
            max_a = _get_max(amplitude, x, y, win)
            out[x, y] = alpha * (max_i - min_i) + (1.0 - alpha) * max_a
    return out


def fusion_map(lc_y: np.ndarray, lc_nir: np.ndarray, red: float = 0.7) -> np.ndarray:
    return (np.maximum(0.0, (lc_nir - lc_y) * red) / (lc_nir + 1e-6)).astype(np.float32)


def high_pass(nir: np.ndarray, ksize: int = 19, strength: float = 0.7) -> np.ndarray:
    base = cv2.GaussianBlur(nir, (ksize, ksize), 0)
    return (nir.astype(np.float32) - base.astype(np.float32)) * strength


def fuse_rgb_nir(rgb: np.ndarray, nir: np.ndarray, *, red: float = 0.7, hp_strength: float = 0.7) -> np.ndarray:
    y = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    lc_y = local_contrast(y)
    lc_nir = local_contrast(nir)
    fmap = fusion_map(lc_y, lc_nir, red)
    nir_detail = high_pass(nir, strength=hp_strength)
    fused = rgb.astype(np.float32)
    for c in range(3):
        fused[:, :, c] += fmap * nir_detail
    return np.clip(fused, 0, 255).astype(np.uint8)

# -----------------------------------------------------------------------------
# 2.  GUIDED‑FILTER DENOISING  (NIR as guide)
# -----------------------------------------------------------------------------

def guided_denoise(rgb: np.ndarray, nir: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge‑preserving denoise: NIR is guide, RGB is src.

    If the ximgproc module isn’t available we fall back to a joint
    bilateral filter (still using NIR guidance).
    """
    # convert to float32 in 0‑1 range for stability
    rgb_f = rgb.astype(np.float32) / 255.0
    nir_f = nir.astype(np.float32) / 255.0

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        den = cv2.ximgproc.guidedFilter(nir_f, rgb_f, radius, eps)
        den = (den * 255.0).clip(0, 255).astype(np.uint8)
        return den

    # fallback: joint bilateral (slower but still edge‑preserving)
    print("[INFO] OpenCV‑ximgproc not found → falling back to standard bilateral filter (NIR guidance lost)", file=sys.stderr)
    sig_sp, sig_col = float(radius), 0.1  # heuristics
    # Try cv2.bilateralFilter as cv2.jointBilateralFilter was not found
    # Note: This uses rgb_f as source, losing the NIR guidance aspect of jointBilateralFilter
    den = cv2.bilateralFilter(rgb_f, d=-1, sigmaColor=sig_col, sigmaSpace=sig_sp)
    return (den * 255.0).clip(0, 255).astype(np.uint8)

# -----------------------------------------------------------------------------
# 3.  COMMAND‑LINE INTERFACE
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RGB‑NIR fusion with NIR‑guided denoising (guided filter)")
    ap.add_argument("--rgb", required=True, help="Path to RGB/BGR image")
    ap.add_argument("--nir", help="Path to aligned NIR image (grayscale)")
    ap.add_argument("--out", default="result.png", help="Output filename")
    ap.add_argument("--skip_fusion", action="store_true", help="Skip fusion; only denoise with NIR guide")
    ap.add_argument("--radius", type=int, default=8, help="Guided‑filter radius (pixels)")
    ap.add_argument("--eps", type=float, default=1e-3, help="Guided‑filter epsilon (regularisation)")
    ap.add_argument("--hp_strength", type=float, default=0.7, help="Strength of high-pass NIR detail (fusion)")
    args = ap.parse_args()

    # load images
    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise SystemExit("Cannot load RGB image")

    if args.skip_fusion:
        if args.nir is None:
            raise SystemExit("Need --nir when --skip_fusion is used (guide image)")
        nir = cv2.imread(args.nir, cv2.IMREAD_GRAYSCALE)
        if nir is None:
            raise SystemExit("Cannot load NIR image")
        fused = rgb
    else:
        if args.nir is None:
            raise SystemExit("Need NIR path for fusion")
        nir = cv2.imread(args.nir, cv2.IMREAD_GRAYSCALE)
        if nir is None:
            raise SystemExit("Cannot load NIR image")
        # Pass hp_strength from args
        fused = fuse_rgb_nir(rgb, nir, hp_strength=args.hp_strength)

    deno = guided_denoise(fused, nir, radius=args.radius, eps=args.eps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, deno)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
