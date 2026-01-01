#!/usr/bin/env python3
"""
Bonus: Overlay sound source directions (DOA) on a blank camera canvas.

Adds a top-left legend:
Source1/Source2 DOA values (azimuth/elevation) in radians.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
import cv2


@dataclass(frozen=True)
class DOA:
    azimuth: float
    elevation: float


# ------------------------- YAML / Camera parsing -------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_camera_params(d: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Returns:
      K: (3,3)
      R: (3,3)
      t: (3,1)
      dist: (N,)
      (W,H)
    """
    K = np.asarray(d["camera_intrinsics"], dtype=np.float64)

    extr = np.asarray(d["camera_extrinsics"], dtype=np.float64)  # (3,4)
    if extr.shape != (3, 4):
        raise ValueError(f"camera_extrinsics expected (3,4), got {extr.shape}")
    R = extr[:, :3]
    t = extr[:, 3:4]  # (3,1)

    dist = np.asarray(d.get("camera_distort", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)

    res = d.get("camera_resolution", [1920.0, 1080.0])
    W = int(round(float(res[0])))
    H = int(round(float(res[1])))

    return K, R, t, dist, (W, H)


def ensure_canvas(blank: bool, W: int, H: int) -> np.ndarray:
    if not blank:
        raise ValueError("No image provided. Use --blank to generate a canvas.")
    return np.zeros((H, W, 3), dtype=np.uint8)


def effective_world_to_cam(R: np.ndarray, t: np.ndarray, invert_extrinsics: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R_eff, t_eff) such that:
        X_cam = R_eff * X_world + t_eff

    Default assumes provided extrinsics are world->camera: Xc = R Xw + t

    If invert_extrinsics=True, treat provided extrinsics as camera->world:
        X_world = R X_cam + t
    Then invert:
        X_cam = R^T (X_world - t) = (R^T) X_world + (-R^T t)
    """
    if not invert_extrinsics:
        return R, t
    R_eff = R.T
    t_eff = -R.T @ t
    return R_eff, t_eff


# ------------------------- DOA utilities -------------------------

def doa_to_unit_vector(doa: DOA) -> np.ndarray:
    """
    Assignment convention:
      x = cos(-az) cos(-el)
      y = sin(-az) cos(-el)
      z = sin(-el)
    """
    az = doa.azimuth
    el = doa.elevation
    u = np.array([
        np.cos(-az) * np.cos(-el),
        np.sin(-az) * np.cos(-el),
        np.sin(-el),
    ], dtype=np.float64)
    u /= (np.linalg.norm(u) + 1e-12)
    return u


# ------------------------- Projection helpers -------------------------

def project_camera_points(K: np.ndarray, dist: np.ndarray, Pc: np.ndarray) -> np.ndarray:
    """
    Project points already in CAMERA coordinates (Pc) to pixels.
    Uses rvec=0, tvec=0.
    """
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    Pc_cv = Pc.reshape(-1, 1, 3).astype(np.float64)
    img_pts, _ = cv2.projectPoints(Pc_cv, rvec, tvec, K.astype(np.float64), dist.astype(np.float64))
    return img_pts.reshape(-1, 2)


def project_world_points(K: np.ndarray, dist: np.ndarray, R_eff: np.ndarray, t_eff: np.ndarray, Pw: np.ndarray) -> np.ndarray:
    """
    Project WORLD points using OpenCV with effective world->camera R_eff,t_eff.
    """
    rvec, _ = cv2.Rodrigues(R_eff.astype(np.float64))
    tvec = t_eff.reshape(3, 1).astype(np.float64)

    Pw_cv = Pw.reshape(-1, 1, 3).astype(np.float64)
    img_pts, _ = cv2.projectPoints(Pw_cv, rvec, tvec, K.astype(np.float64), dist.astype(np.float64))
    return img_pts.reshape(-1, 2)


# ------------------------- Drawing helpers -------------------------

def draw_principal_point(img: np.ndarray, cx: int, cy: int) -> None:
    cv2.drawMarker(img, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


def draw_label(img: np.ndarray, xy: Tuple[int, int], text: str) -> None:
    x, y = xy
    cv2.circle(img, (x, y), 12, (0, 255, 255), 2)
    cv2.putText(img, text, (x + 14, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)


def intersect_ray_with_image_border(
    cx: float, cy: float, x: float, y: float, W: int, H: int, margin: int = 20
) -> Tuple[int, int]:
    dx = x - cx
    dy = y - cy

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return int(round(cx)), int(round(cy))

    t_candidates = []

    if dx != 0:
        t_left = (margin - cx) / dx
        t_right = ((W - 1 - margin) - cx) / dx
        if t_left > 0:
            t_candidates.append(t_left)
        if t_right > 0:
            t_candidates.append(t_right)

    if dy != 0:
        t_top = (margin - cy) / dy
        t_bottom = ((H - 1 - margin) - cy) / dy
        if t_top > 0:
            t_candidates.append(t_top)
        if t_bottom > 0:
            t_candidates.append(t_bottom)

    if not t_candidates:
        bx = min(max(int(round(x)), margin), W - 1 - margin)
        by = min(max(int(round(y)), margin), H - 1 - margin)
        return bx, by

    t = min(t_candidates)
    bx = cx + t * dx
    by = cy + t * dy

    bx = min(max(int(round(bx)), margin), W - 1 - margin)
    by = min(max(int(round(by)), margin), H - 1 - margin)
    return bx, by


def draw_offscreen_arrow(img: np.ndarray, cx: int, cy: int, x: int, y: int, W: int, H: int, label: str) -> None:
    bx, by = intersect_ray_with_image_border(cx, cy, x, y, W, H, margin=20)
    cv2.arrowedLine(img, (cx, cy), (bx, by), (0, 255, 255), 2, tipLength=0.08)
    draw_label(img, (bx, by), label + " (offscreen)")


def draw_top_left_legend(img: np.ndarray, lines: list[str], x0: int = 20, y0: int = 30) -> None:
    """
    Draw a small readable legend box in the top-left corner.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_h = 28

    # Measure box size
    widths = []
    for s in lines:
        (w, h), _ = cv2.getTextSize(s, font, font_scale, thickness)
        widths.append(w)
    box_w = max(widths) + 20
    box_h = line_h * len(lines) + 18

    # Background rectangle (dark)
    cv2.rectangle(img, (x0 - 10, y0 - 22), (x0 - 10 + box_w, y0 - 22 + box_h), (20, 20, 20), -1)
    # Border (yellow-ish)
    cv2.rectangle(img, (x0 - 10, y0 - 22), (x0 - 10 + box_w, y0 - 22 + box_h), (0, 255, 255), 2)

    # Text
    for i, s in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(img, s, (x0, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


# ------------------------- Main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Path to array_geometry.yaml")
    ap.add_argument("--blank", action="store_true", help="Create blank canvas using camera_resolution")
    ap.add_argument("--out", default="outputs/sound_overlay.png", help="Output image path")
    ap.add_argument("--distance", type=float, default=5.0, help="Distance along ray (meters)")
    ap.add_argument("--show_mics", action="store_true", help="Also project and draw microphone positions")
    ap.add_argument("--invert_extrinsics", action="store_true",
                    help="Treat extrinsics as camera->world instead of world->camera")
    args = ap.parse_args()

    d = load_yaml(args.yaml)
    K, R, t, dist, (W, H) = parse_camera_params(d)
    img = ensure_canvas(args.blank, W, H)

    # Principal point from intrinsics
    cx = int(round(float(K[0, 2])))
    cy = int(round(float(K[1, 2])))

    draw_principal_point(img, cx, cy)

    # DOAs from assignment (EXACT as requested)
    doa1 = DOA(azimuth=-0.069, elevation=0.0)
    doa2 = DOA(azimuth=1.029, elevation=0.017)

    # Draw top-left legend (exact formatting)
    legend_lines = [
        "Source1: azimuth = -0.069 rad, elevation = 0 rad",
        "Source2: azimuth = 1.029 rad, elevation = 0.017 rad",
    ]
    draw_top_left_legend(img, legend_lines)

    sources = [
        ("Source1", doa1),
        ("Source2", doa2),
    ]

    # Effective world->camera transform
    R_eff, t_eff = effective_world_to_cam(R, t, args.invert_extrinsics)

    for name, doa in sources:
        u_w = doa_to_unit_vector(doa)
        u_c = (R_eff @ u_w.reshape(3, 1)).reshape(3)

        # Ensure ray goes forward in camera frame (Z > 0)
        if u_c[2] <= 1e-6:
            u_c = -u_c

        P_c = (args.distance * u_c).reshape(1, 3)
        uv = project_camera_points(K, dist, P_c)[0]

        x = int(round(float(uv[0])))
        y = int(round(float(uv[1])))
        inside = (0 <= x < W) and (0 <= y < H)

        print(f"[INFO] {name}: pixel=({x},{y}) inside={inside}")

        if inside:
            draw_label(img, (x, y), name)
        else:
            draw_offscreen_arrow(img, cx, cy, x, y, W, H, name)

    # Optional: draw microphones (world points)
    if args.show_mics and "array_geometry" in d:
        mics_w = np.asarray(d["array_geometry"], dtype=np.float64)  # (M,3)
        uv_m = project_world_points(K, dist, R_eff, t_eff, mics_w)
        for (u, v) in uv_m:
            mx = int(round(float(u)))
            my = int(round(float(v)))
            if 0 <= mx < W and 0 <= my < H:
                cv2.circle(img, (mx, my), 2, (255, 0, 0), -1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[DONE] Wrote overlay image: {out_path}")


if __name__ == "__main__":
    main()
