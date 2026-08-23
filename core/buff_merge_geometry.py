# -*- coding: utf-8 -*-
"""能量机关轮廓合并与极轴极值点计算"""
import numpy as np
import cv2


def merge_contour_polys(poly_list):
    """将多个 SAM 轮廓点合并为统一点集。"""
    if not poly_list:
        return None
    pts = []
    for poly in poly_list:
        if not poly:
            continue
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(arr) > 0:
            pts.append(arr)
    if not pts:
        return None
    return np.vstack(pts)


def merge_contours_via_mask(polys, img_w, img_h):
    """通过 mask 合并多轮廓，返回最大外轮廓点集。"""
    points = merge_contour_polys(polys)
    if points is None:
        return None

    mask = np.zeros((max(int(img_h), 1), max(int(img_w), 1)), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 3:
            continue
        cnt = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [cnt], 255)

    if mask.max() == 0:
        return points

    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    except AttributeError:
        pass

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return points

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return points
    return largest.reshape(-1, 2).astype(np.float64)


def compute_axis_extremes(points):
    """
    基于 PCA 计算图案相对极轴上的四个极值点（长轴两端 + 短轴两端）。
    返回 [(x,y), ...] 共 4 个点。
    """
    if points is None or len(points) < 3:
        return None

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    centered = pts - center

    if len(pts) >= 5:
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            return _fallback_min_area_extremes(pts)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        long_axis = eigenvectors[:, order[0]]
        short_axis = eigenvectors[:, order[1]]
    else:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box = cv2.boxPoints(rect)
        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        if np.linalg.norm(edge1) >= np.linalg.norm(edge2):
            long_axis = edge1 / (np.linalg.norm(edge1) + 1e-8)
            short_axis = edge2 / (np.linalg.norm(edge2) + 1e-8)
        else:
            long_axis = edge2 / (np.linalg.norm(edge2) + 1e-8)
            short_axis = edge1 / (np.linalg.norm(edge1) + 1e-8)

    def axis_endpoints(axis):
        proj = centered @ axis
        return (
            center + proj.min() * axis,
            center + proj.max() * axis,
        )

    long_neg, long_pos = axis_endpoints(long_axis)
    short_neg, short_pos = axis_endpoints(short_axis)
    return [
        (float(long_neg[0]), float(long_neg[1])),
        (float(long_pos[0]), float(long_pos[1])),
        (float(short_neg[0]), float(short_neg[1])),
        (float(short_pos[0]), float(short_pos[1])),
    ]


def _fallback_min_area_extremes(pts):
    rect = cv2.minAreaRect(pts.astype(np.float32))
    box = cv2.boxPoints(rect)
    return [(float(p[0]), float(p[1])) for p in box]


def refine_keypoints_with_extremes(yolo_kpts, extremes, fixed_index=1):
    """
    将第 1、3、4、5 个关键点（1-indexed）对齐到极轴极值点，第 2 个保持不变。
    yolo_kpts: [[x,y,v], ...] 至少 5 个
    """
    if not extremes or len(extremes) < 4:
        return yolo_kpts

    adjust_indices = [0, 2, 3, 4]
    new_kpts = [list(k) for k in yolo_kpts]
    available = list(extremes)

    for idx in adjust_indices:
        if idx >= len(new_kpts):
            continue
        yx, yy = new_kpts[idx][0], new_kpts[idx][1]
        best_j = min(
            range(len(available)),
            key=lambda j: (available[j][0] - yx) ** 2 + (available[j][1] - yy) ** 2,
        )
        ex, ey = available.pop(best_j)
        new_kpts[idx][0] = float(ex)
        new_kpts[idx][1] = float(ey)

    if fixed_index < len(new_kpts) and len(yolo_kpts) > fixed_index:
        new_kpts[fixed_index][0] = float(yolo_kpts[fixed_index][0])
        new_kpts[fixed_index][1] = float(yolo_kpts[fixed_index][1])
        new_kpts[fixed_index][2] = yolo_kpts[fixed_index][2]

    return new_kpts


def compute_circle_keypoints(polys, yolo_kpts, bbox):
    """
    合并 SAM 轮廓 → PCA 极值点 → 按最近距离分配给 adjust 关键点，跳过固定中心点。
    返回 [(x,y), ...]，长度等于 yolo_kpts，第0位（固定中心点）保持原坐标。
    """
    img_w = bbox.width()
    img_h = bbox.height()
    ox, oy = bbox.x(), bbox.y()

    local_polys = [[(px - ox, py - oy) for px, py in poly] for poly in polys]

    merged = merge_contours_via_mask(local_polys, img_w, img_h)
    if merged is None:
        merged = merge_contour_polys(local_polys)
    if merged is None:
        return None

    merged_scene = merged + np.array([ox, oy])

    extremes = compute_axis_extremes(merged_scene)
    if not extremes:
        return None

    FIXED = 0
    adjust = [1, 2, 3, 4]
    result = [(float(k[0]), float(k[1])) for k in yolo_kpts]
    available = list(extremes)
    for idx in adjust:
        if idx >= len(result) or not available:
            break
        yx, yy = yolo_kpts[idx][0], yolo_kpts[idx][1]
        best_j = min(range(len(available)),
                     key=lambda j: (available[j][0] - yx) ** 2 + (available[j][1] - yy) ** 2)
        result[idx] = available.pop(best_j)

    return result
