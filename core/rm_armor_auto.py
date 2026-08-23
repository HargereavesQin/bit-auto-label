# -*- coding: utf-8 -*-
"""
RM装甲板半自动标注模块
在Pose模式下，每画一个bbox（手动），自动使用SAM文字提示"light bar"检测装甲板灯条，
从返回的2个灯条mask中提取4个极值点（左上、左下、右下、右上），
自动标注为关键点。

新版改进：
1. 椭圆拟合：使用 cv2.fitEllipse 拟合灯条轮廓，取长轴（主轴）两端点作为关键点
2. 关键点限幅：所有关键点强制在 bbox 区域内
3. 数字识别 + 边界框收缩：发送 "number" 提示词，将 bbox 收缩到与灯条椭圆和数字 mask 内切
"""
from PySide6.QtCore import QObject, Signal, QPointF
from PySide6.QtCore import QRectF
import numpy as np
import cv2
import math


class RMArmorAuto(QObject):
    """RM装甲板半自动标注控制器"""

    keypoints_placed = Signal(object, list)
    error_message = Signal(str)
    full_auto_step_placed = Signal(object, list)
    full_auto_regions_ready = Signal(list)  # [(cx, cy, bw, bh, search_radius), ...]
    full_auto_finished = Signal()  # 全自动周期完成（成功或失败）

    VERTICAL_ANGLE_THRESHOLD = 50.0
    PARALLEL_THRESHOLD = 45.0
    BAR_LENGTH_MIN_RATIO = 0.5  # 灯条端点距离阈值 = max(num_w,num_h) * BAR_LENGTH_MIN_RATIO
    SEARCH_RADIUS_RATIO = 1.5  # 灯条搜索半径 = max(num_w,num_h) * SEARCH_RADIUS_RATIO
    ROI_PADDING = 10  # ROI 边距像素

    def __init__(self, sam_client, parent=None):
        super().__init__(parent)
        self.sam_client = sam_client
        self.enabled = False
        self.pending_shape = None
        self._phase = "idle"
        self._bar_ellipses = []
        self._left_bar_pts = None
        self._right_bar_pts = None
        self._number_center = None
        self._full_auto_category = None
        self._full_auto_numbers = []
        self._full_auto_index = 0
        self.sam_client.text_result_ready.connect(self._on_text_result)

    def set_enabled(self, enabled):
        self.enabled = enabled

    def request_auto_keypoints(self, pose_shape):
        if not self.enabled:
            return
        if not self.sam_client.model:
            self._emit_failure("SAM模型未就绪")
            return
        if not self.sam_client.inference_worker.inference_state:
            self._emit_failure("SAM图像特征未就绪，请等待图片特征提取完成")
            return
        self.pending_shape = pose_shape
        self._phase = "number_first"
        self._bar_ellipses = []
        self._left_bar_pts = None
        self._right_bar_pts = None
        self._number_center = None
        self.sam_client.request_text_inference("number")

    def _on_text_result(self, results, prompt_text):
        print(f"  [rm_auto] _on_text_result phase={self._phase} prompt={prompt_text} results={len(results)}")
        parent = self.parent()
        if parent is not None and hasattr(parent, 'rm_buff_merge_auto'):
            if parent.rm_buff_merge_auto.phase != "idle":
                return
        if self._phase == "full_auto_number" and prompt_text == "number":
            self._handle_full_auto_number_results(results)
            return
        elif self._phase == "full_auto_light_bar" and prompt_text == "light":
            self._handle_full_auto_light_bar_results(results)
            return

        if not self.enabled:
            print("  [rm_auto] _on_text_result: not enabled, skip")
            return

        pose_rect = self.pending_shape.rect() if self.pending_shape else None

        if self._phase == "number_first" and prompt_text == "number":
            self._handle_number_first_results(results, pose_rect)
        elif self._phase == "light_bar" and prompt_text == "light":
            self._handle_light_bar_results(results, pose_rect)
        elif self._phase == "number" and prompt_text == "number":
            self._handle_number_results(results, pose_rect)

    def _handle_number_first_results(self, results, pose_rect):
        number_result = None
        for res in results:
            bx, by, bw, bh = res["rect"]
            num_rect = QRectF(bx, by, bw, bh)
            if pose_rect.intersects(num_rect):
                number_result = res
                break

        if number_result:
            pts = number_result["poly_pts"]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                self._number_center = (sum(xs) / len(xs), sum(ys) / len(ys))

        self._phase = "light_bar"
        self.sam_client.request_text_inference("light")

    def _handle_light_bar_results(self, results, pose_rect):
        def bar_center_x(r):
            pts = r["poly_pts"]
            return sum(p[0] for p in pts) / len(pts)

        filtered = []
        for res in results:
            bx, by, bw, bh = res["rect"]
            bar_rect = QRectF(bx, by, bw, bh)
            if pose_rect.intersects(bar_rect):
                filtered.append(res)

        if len(filtered) < 2:
            self._emit_failure("未检测到足够的灯条（< 2）")
            return

        if self._number_center is not None:
            num_cx = self._number_center[0]
            left_candidates = [r for r in filtered if bar_center_x(r) < num_cx]
            right_candidates = [r for r in filtered if bar_center_x(r) > num_cx]
            if left_candidates and right_candidates:
                left_bar = max(left_candidates, key=lambda r: bar_center_x(r))
                right_bar = min(right_candidates, key=lambda r: bar_center_x(r))
            elif left_candidates:
                left_bar = max(left_candidates, key=lambda r: bar_center_x(r))
                right_bar = max([r for r in filtered if r != left_bar], key=lambda r: bar_center_x(r))
            elif right_candidates:
                right_bar = min(right_candidates, key=lambda r: bar_center_x(r))
                left_bar = min([r for r in filtered if r != right_bar], key=lambda r: bar_center_x(r))
            else:
                left_bar = min(filtered, key=bar_center_x)
                right_bar = max(filtered, key=bar_center_x)
        else:
            left_bar = min(filtered, key=lambda r: min(p[0] for p in r["poly_pts"]))
            right_bar = max(filtered, key=lambda r: max(p[0] for p in r["poly_pts"]))

        left_endpoints, left_ellipse = self._fit_ellipse_endpoints(left_bar["poly_pts"])
        right_endpoints, right_ellipse = self._fit_ellipse_endpoints(right_bar["poly_pts"])

        if left_endpoints is None or right_endpoints is None:
            self._emit_failure("椭圆拟合失败")
            return

        dev_left = self._endpoint_deviation_from_vertical(left_endpoints)
        dev_right = self._endpoint_deviation_from_vertical(right_endpoints)
        if dev_left > self.VERTICAL_ANGLE_THRESHOLD:
            self._emit_failure(f"左灯条垂直度不足 (偏差: {dev_left:.1f}°)")
            return
        if dev_right > self.VERTICAL_ANGLE_THRESHOLD:
            self._emit_failure(f"右灯条垂直度不足 (偏差: {dev_right:.1f}°)")
            return

        parallel_angle = self._endpoint_parallelism_angle(left_endpoints, right_endpoints)
        if parallel_angle > self.PARALLEL_THRESHOLD:
            self._emit_failure(f"灯条平行度不足 (夹角: {parallel_angle:.1f}°)")
            return

        self._bar_ellipses = [left_ellipse, right_ellipse]
        self._left_bar_pts = left_endpoints
        self._right_bar_pts = right_endpoints

        if self._number_center is None:
            all_points = []
            for ellipse in self._bar_ellipses:
                box = cv2.boxPoints(ellipse)
                all_points.extend([(float(p[0]), float(p[1])) for p in box])
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                new_rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                shrinked = new_rect.intersected(pose_rect)
                self.pending_shape.setRect(shrinked)
                if hasattr(self.pending_shape, 'update_label_position'):
                    self.pending_shape.update_label_position(self.pending_shape)
            kpts = self._build_keypoints(shrinked if shrinked else pose_rect)
            self.keypoints_placed.emit(self.pending_shape, kpts)
            self._reset()
        else:
            self._phase = "number"
            self.sam_client.request_text_inference("number")

    def _handle_number_results(self, results, pose_rect):
        number_result = None
        for res in results:
            bx, by, bw, bh = res["rect"]
            num_rect = QRectF(bx, by, bw, bh)
            if pose_rect.intersects(num_rect):
                number_result = res
                break

        all_points = []

        for ellipse in self._bar_ellipses:
            box = cv2.boxPoints(ellipse)
            all_points.extend([(float(p[0]), float(p[1])) for p in box])

        if number_result:
            all_points.extend([(float(p[0]), float(p[1])) for p in number_result["poly_pts"]])

        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            new_rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            shrinked = new_rect.intersected(pose_rect)
            self.pending_shape.setRect(shrinked)
            if hasattr(self.pending_shape, 'update_label_position'):
                self.pending_shape.update_label_position(self.pending_shape)
            clamp_rect = shrinked
        else:
            clamp_rect = pose_rect

        kpts = self._build_keypoints(clamp_rect)

        self.keypoints_placed.emit(self.pending_shape, kpts)
        self._reset()

    def _fit_ellipse_endpoints(self, poly_pts):
        pts = np.array(poly_pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) < 5:
            return None, None
        try:
            ellipse = cv2.fitEllipse(pts)
        except Exception:
            return None, None

        cx, cy = ellipse[0]
        size_w, size_h = ellipse[1]
        angle_deg = ellipse[2]

        if size_w >= size_h:
            semi_a = size_w / 2
            theta = math.radians(angle_deg)
        else:
            semi_a = size_h / 2
            theta = math.radians(angle_deg + 90)

        dx = semi_a * math.cos(theta)
        dy = semi_a * math.sin(theta)

        pt1 = (cx + dx, cy + dy)
        pt2 = (cx - dx, cy - dy)

        print(f"  [rm_auto] ellipse: center=({cx:.1f},{cy:.1f}) size=({size_w:.1f},{size_h:.1f}) angle={angle_deg:.1f}")
        print(f"  [rm_auto] endpoints: pt1=({pt1[0]:.2f},{pt1[1]:.2f}) pt2=({pt2[0]:.2f},{pt2[1]:.2f}) dist={math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]):.2f}")

        return (pt1, pt2), ellipse

    def _build_keypoints(self, roi_rect):
        def clamp(pt):
            x = max(roi_rect.left(), min(roi_rect.right(), pt[0]))
            y = max(roi_rect.top(), min(roi_rect.bottom(), pt[1]))
            return (x, y)

        left_clamped = [clamp(self._left_bar_pts[0]), clamp(self._left_bar_pts[1])]
        right_clamped = [clamp(self._right_bar_pts[0]), clamp(self._right_bar_pts[1])]

        l_top, l_bot = min(left_clamped, key=lambda p: p[1]), max(left_clamped, key=lambda p: p[1])
        r_top, r_bot = min(right_clamped, key=lambda p: p[1]), max(right_clamped, key=lambda p: p[1])

        kpts = [
            [l_top[0], l_top[1], 2],
            [l_bot[0], l_bot[1], 2],
            [r_bot[0], r_bot[1], 2],
            [r_top[0], r_top[1], 2],
        ]
        return kpts

    def _endpoint_deviation_from_vertical(self, endpoints):
        pt1, pt2 = endpoints
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        angle_deg = math.degrees(math.atan2(abs(dy), abs(dx)))
        return abs(90 - angle_deg)

    def _endpoint_parallelism_angle(self, left_ends, right_ends):
        def direction(endpoints):
            dx = endpoints[1][0] - endpoints[0][0]
            dy = endpoints[1][1] - endpoints[0][1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1:
                return (0.0, 0.0)
            return (dx / length, dy / length)

        lx, ly = direction(left_ends)
        rx, ry = direction(right_ends)
        if (lx == 0 and ly == 0) or (rx == 0 and ry == 0):
            return 90.0
        dot = abs(lx * rx + ly * ry)
        dot = min(1.0, dot)
        return math.degrees(math.acos(dot))

    def _pick_closest_valid_bar(self, candidates, ref_x, ref_y, min_length=0):
        sorted_candidates = sorted(candidates, key=lambda t: math.hypot(t[0] - ref_x, t[1] - ref_y))
        for bcx, bcy, res in sorted_candidates:
            endpoints, ellipse = self._fit_ellipse_endpoints(res["poly_pts"])
            if endpoints is None:
                print(f"  [rm_auto] bar ({bcx:.1f},{bcy:.1f}): fitEllipse failed, skip")
                continue
            dx = endpoints[1][0] - endpoints[0][0]
            dy = endpoints[1][1] - endpoints[0][1]
            dist = math.hypot(dx, dy)
            if dist < 5:
                print(f"  [rm_auto] bar ({bcx:.1f},{bcy:.1f}): endpoint dist {dist:.2f}px < 5, skip")
                continue
            if min_length > 0 and dist < min_length:
                print(f"  [rm_auto] bar ({bcx:.1f},{bcy:.1f}): endpoint dist {dist:.2f}px < min_length {min_length:.2f}px, skip")
                continue
            dev = self._endpoint_deviation_from_vertical(endpoints)
            if dev > self.VERTICAL_ANGLE_THRESHOLD:
                print(f"  [rm_auto] bar ({bcx:.1f},{bcy:.1f}): deviation {dev:.1f}deg > {self.VERTICAL_ANGLE_THRESHOLD}deg, skip")
                continue
            return (bcx, bcy, res), endpoints, ellipse, dev
        return None

    def _emit_failure(self, msg):
        self.error_message.emit(msg)
        self._reset()

    def request_full_auto(self, category):
        if self._phase != "idle":
            # 防止重复点击造成 _full_auto_numbers 被清空, 状态错位
            self.error_message.emit(f"全自动标注进行中 (phase={self._phase}), 请先等待当前周期完成")
            return
        if not self.sam_client.model:
            self.error_message.emit("SAM模型未就绪")
            return
        if not self.sam_client.inference_worker.inference_state:
            self.error_message.emit("SAM图像特征未就绪，请等待加载完成")
            return
        self._full_auto_category = category
        self._full_auto_numbers = []
        self._full_auto_index = 0
        self._phase = "full_auto_number"
        self.sam_client.request_text_inference("number")

    def _handle_full_auto_number_results(self, results):
        self._full_auto_numbers = []
        for res in results:
            pts = res["poly_pts"]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = sum(xs) / len(pts)
                cy = sum(ys) / len(pts)
                bw = res["rect"][2]
                bh = res["rect"][3]
                self._full_auto_numbers.append((cx, cy, bw, bh, pts))
        print(f"  [rm_auto] detected {len(self._full_auto_numbers)} number(s): "
              + ", ".join(f"({cx:.0f},{cy:.0f}) bbox={bw:.0f}x{bh:.0f}" for cx, cy, bw, bh, pts in self._full_auto_numbers))
        if not self._full_auto_numbers:
            self.error_message.emit("未检测到数字")
            self._reset_full_auto()
            return
        regions = [(cx, cy, bw, bh, max(bw, bh) * self.SEARCH_RADIUS_RATIO) for cx, cy, bw, bh, pts in self._full_auto_numbers]
        self.full_auto_regions_ready.emit(regions)
        self._phase = "full_auto_light_bar"
        self.sam_client.request_text_inference("light")

    def _handle_full_auto_light_bar_results(self, results):
        if not self._full_auto_numbers:
            self._reset_full_auto()
            return
        for num_cx, num_cy, num_w, num_h, num_poly_pts in self._full_auto_numbers:
            search_radius = max(num_w, num_h) * self.SEARCH_RADIUS_RATIO

            bar_data = []
            for res in results:
                pts = res["poly_pts"]
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bcx = sum(xs) / len(pts)
                    bcy = sum(ys) / len(pts)
                    dist = math.hypot(bcx - num_cx, bcy - num_cy)
                    if dist <= search_radius:
                        bar_data.append((bcx, bcy, res))

            if len(bar_data) < 2:
                print(f"  [rm_auto] number at ({num_cx:.0f},{num_cy:.0f}): only {len(bar_data)} bar(s) within radius {search_radius:.0f}, skip")
                continue

            left_candidates = [(bcx, bcy, res) for bcx, bcy, res in bar_data if bcx < num_cx]
            right_candidates = [(bcx, bcy, res) for bcx, bcy, res in bar_data if bcx > num_cx]
            if not left_candidates or not right_candidates:
                print(f"  [rm_auto] number at ({num_cx:.0f},{num_cy:.0f}): no left or right bar within {search_radius:.0f}px, skip")
                continue

            number_ref_dist = max(num_w, num_h) * self.BAR_LENGTH_MIN_RATIO
            left_result = self._pick_closest_valid_bar(left_candidates, num_cx, num_cy, min_length=number_ref_dist)
            right_result = self._pick_closest_valid_bar(right_candidates, num_cx, num_cy, min_length=number_ref_dist)
            if left_result is None:
                print(f"  [rm_auto] number at ({num_cx:.0f},{num_cy:.0f}): no valid left bar (fit/degenerate/threshold), skip")
                continue
            if right_result is None:
                print(f"  [rm_auto] number at ({num_cx:.0f},{num_cy:.0f}): no valid right bar (fit/degenerate/threshold), skip")
                continue
            left_bar_data, left_endpoints, left_ellipse, dev_left = left_result
            right_bar_data, right_endpoints, right_ellipse, dev_right = right_result

            print(f"  [rm_auto] selected left:  endpoints=({left_endpoints[0][0]:.1f},{left_endpoints[0][1]:.1f})-({left_endpoints[1][0]:.1f},{left_endpoints[1][1]:.1f}) dev={dev_left:.1f}deg")
            print(f"  [rm_auto] selected right: endpoints=({right_endpoints[0][0]:.1f},{right_endpoints[0][1]:.1f})-({right_endpoints[1][0]:.1f},{right_endpoints[1][1]:.1f}) dev={dev_right:.1f}deg")

            parallel_angle = self._endpoint_parallelism_angle(left_endpoints, right_endpoints)
            if parallel_angle > self.PARALLEL_THRESHOLD:
                print(f"  [rm_auto] parallel angle {parallel_angle:.1f}deg > {self.PARALLEL_THRESHOLD}deg, skip")
                continue

            all_points = []
            for ellipse in [left_ellipse, right_ellipse]:
                box = cv2.boxPoints(ellipse)
                all_points.extend([(float(p[0]), float(p[1])) for p in box])
            if num_poly_pts:
                all_points.extend([(float(p[0]), float(p[1])) for p in num_poly_pts])

            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                from PySide6.QtCore import QRectF as QRectFClass
                roi_rect = QRectFClass(x_min, y_min, x_max - x_min, y_max - y_min)

            left_sorted = sorted(left_endpoints, key=lambda p: p[1])
            right_sorted = sorted(right_endpoints, key=lambda p: p[1])
            l_top, l_bot = left_sorted[0], left_sorted[1]
            r_top, r_bot = right_sorted[0], right_sorted[1]
            kpts = [
                [l_top[0], l_top[1], 2],
                [l_bot[0], l_bot[1], 2],
                [r_bot[0], r_bot[1], 2],
                [r_top[0], r_top[1], 2],
            ]
            print(f"  [rm_auto] EMIT: number_ref_dist={number_ref_dist:.1f}px (num={max(num_w,num_h):.0f}px * ratio={self.BAR_LENGTH_MIN_RATIO})")
            print(f"  [rm_auto] EMIT kpts: l_top=({l_top[0]:.2f},{l_top[1]:.2f}) l_bot=({l_bot[0]:.2f},{l_bot[1]:.2f}) r_bot=({r_bot[0]:.2f},{r_bot[1]:.2f}) r_top=({r_top[0]:.2f},{r_top[1]:.2f})")
            print(f"  [rm_auto] EMIT kpts list: {kpts}")
            self.full_auto_step_placed.emit((roi_rect, self._full_auto_category), kpts)
        self._reset_full_auto()

    def _reset_full_auto(self, emit_finished=True):
        self._full_auto_category = None
        self._full_auto_numbers = []
        self._full_auto_index = 0
        self._phase = "idle"
        if emit_finished:
            self.full_auto_finished.emit()

    def _reset(self):
        self.pending_shape = None
        self._phase = "idle"
        self._bar_ellipses = []
        self._left_bar_pts = None
        self._right_bar_pts = None
        self._number_center = None
        # 半自动结束不得误发 full_auto_finished，否则批量状态机会提前跳图
        self._reset_full_auto(emit_finished=False)