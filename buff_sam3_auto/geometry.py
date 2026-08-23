"""双轮廓极轴几何：R 轮廓 + 圆盘轮廓 → 五点。

思路
----
1. 从 SAM3 的 R mask 取连通分量质心作为极点（关键点 0）
2. 从圆盘 mask 选出外轮廓（形态学开运算切断流水灯条）
3. 在圆盘轮廓上：
   - 距极点最近 / 最远 → 极轴（长轴）两端（关键点 1 / 3）
   - 垂直于极轴的最长轮廓弦 → 短轴两端（关键点 2 / 4）

五点语义（0-based）
------------------
0: R 标志几何中心（极点）
1: 圆盘靠近 R 的长轴端
2: 圆盘短轴正向端（图像下侧 / 6 点钟）
3: 圆盘距 R 最远的长轴端
4: 圆盘短轴负向端（图像上侧 / 12 点钟）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _component_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return [c for c in contours if len(c) >= 12 and cv2.contourArea(c) > 0]


def select_r_component(mask: np.ndarray) -> Optional[np.ndarray]:
    """R 通常是较小实心区域；剔除极小噪声后取面积最大的分量。"""
    contours = _component_contours(mask)
    if not contours:
        return None
    area_limit = mask.shape[0] * mask.shape[1] * 0.25
    valid = [c for c in contours if cv2.contourArea(c) <= area_limit]
    return max(valid or contours, key=cv2.contourArea)


def contour_centroid(contour: np.ndarray) -> Optional[np.ndarray]:
    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-6:
        return None
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )


def select_circle_contour(
    circle_mask: np.ndarray, r_mask: Optional[np.ndarray], r_center: np.ndarray
) -> Optional[np.ndarray]:
    """选择圆盘外轮廓，并用形态学开运算切断细长流水灯条。"""
    clean = circle_mask.copy()
    if r_mask is not None:
        radius = max(2, int(round(min(clean.shape[:2]) * 0.015)))
        kernel_r = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        clean[cv2.dilate(r_mask, kernel_r) > 0] = 0

    # 流水灯条细而长；小尺度开运算优先断开它与圆盘的连接。
    k = max(5, int(round(min(clean.shape[:2]) * 0.06)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    contours = _component_contours(opened)
    if not contours:
        contours = _component_contours(clean)
    if not contours:
        return None

    image_area = float(clean.shape[0] * clean.shape[1])
    diag = float(np.hypot(*clean.shape[:2]))
    best, best_score = None, -np.inf
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.002 or area > image_area * 0.80:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = float(
            np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0, 1)
        )
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1e-6)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        if aspect > 3.2 or solidity < 0.30:
            continue
        center = contour_centroid(contour)
        if center is None:
            continue
        distance = float(np.linalg.norm(center - r_center) / max(diag, 1.0))
        size_score = float(np.sqrt(area / image_area))
        score = 2.0 * circularity + solidity + 0.8 * distance + 0.3 * size_score
        if score > best_score:
            best, best_score = contour, score
    return best


def _longest_perpendicular_chord(
    contour_points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    bin_width: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """O(N log N) 搜索垂直于长轴的最长轮廓弦。"""
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    rel = contour_points - origin
    u = rel @ axis
    v = rel @ normal
    bins = np.rint(u / max(bin_width, 1.0)).astype(np.int32)
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1]
    ends = np.r_[starts[1:], len(order)]
    if len(starts) == 0:
        return None

    best_span = -1.0
    best_pair = None
    for start, end in zip(starts, ends):
        ids = order[start:end]
        if len(ids) < 2:
            continue
        local_v = v[ids]
        lo = ids[int(np.argmin(local_v))]
        hi = ids[int(np.argmax(local_v))]
        span = float(v[hi] - v[lo])
        if span > best_span:
            best_span = span
            best_pair = (contour_points[hi], contour_points[lo])
    return best_pair


def solve_five_keypoints(
    r_center: Sequence[float], circle_contour: np.ndarray
) -> Optional[List[Tuple[float, float, int]]]:
    """由 R 中心（极点）和圆盘轮廓求五点，复杂度 O(N log N)。"""
    p0 = np.asarray(r_center, dtype=np.float32)
    points = circle_contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 12:
        return None

    dist2 = np.einsum("ij,ij->i", points - p0, points - p0)
    p1 = points[int(np.argmin(dist2))]  # 靠近 R 的长轴端
    p3 = points[int(np.argmax(dist2))]  # 距 R 最远的长轴端
    axis_raw = p3 - p0
    norm = float(np.linalg.norm(axis_raw))
    if norm < 2.0:
        return None
    axis = axis_raw / norm

    x, y, w, h = cv2.boundingRect(circle_contour)
    chord = _longest_perpendicular_chord(
        points, p0, axis, max(1.0, min(w, h) / 128.0)
    )
    if chord is None:
        return None
    side_a, side_b = chord
    # 索引2为图像下侧(6点钟)，索引4为上侧(12点钟)。
    p2, p4 = (
        (side_a, side_b) if side_a[1] >= side_b[1] else (side_b, side_a)
    )
    return [
        (float(p0[0]), float(p0[1]), 2),
        (float(p1[0]), float(p1[1]), 2),
        (float(p2[0]), float(p2[1]), 2),
        (float(p3[0]), float(p3[1]), 2),
        (float(p4[0]), float(p4[1]), 2),
    ]
