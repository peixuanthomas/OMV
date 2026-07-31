"""One-time red-A4 calibration for the fixed OpenMV H7 Plus camera.

Calibration temporarily uses RGB565 to isolate the red sheet. Production
recognition remains VGA grayscale. Keep the A4 long edge pointing right and
the short edge pointing down. The detected frame and named corners are drawn
throughout the calibration so the operator can confirm the correct target.
"""

import gc
import time

import csi

import calibration_geometry as calibration
import polygon_geometry as geometry
from polygon_detection import (
    _blob_perimeter,
    _blob_pixels,
    _blob_rect,
    _find_blob_boundary_seed,
    trace_outer_boundary,
)


# Broad LAB threshold for red paper. If lighting is unusual, tune this single
# constant in OpenMV IDE's Threshold Editor; the gray table should remain out.
RED_THRESHOLD = (5, 100, 15, 127, -20, 127)

FRAME_BUFFER_COUNT = 1
CAMERA_WARMUP_MS = 1800
PAPER_MARGIN_PX = 12
MIN_PAPER_AREA_PX = 42000
MIN_PAPER_PIXELS = 28000
MAX_CONTOUR_POINTS = 9000
MAX_OVERLAY_CONTOUR_SEGMENTS = 180
MAX_CORNER_REFINEMENT_SHIFT_PX = 14.0

STABLE_DETECTIONS = 6
CANDIDATE_FRAMES = 2
FINAL_VALIDATION_FRAMES = 60

LENS_STRENGTH_MIN = 0.6
LENS_STRENGTH_MAX = 3.0
LENS_STRENGTH_COARSE_STEP = 0.2
OPTICAL_CENTER_RANGE_PX = 16
OPTICAL_CENTER_COARSE_STEP_PX = 4

OVERLAY_SHADOW = (0, 0, 0)
OVERLAY_FRAME = (0, 255, 0)
OVERLAY_CORNER = (255, 255, 0)
OVERLAY_ERROR = (255, 0, 255)
OVERLAY_TEXT = (255, 255, 255)


def _draw_text(frame, x, y, text, color=OVERLAY_TEXT):
    frame.draw_string(
        (x + 1, y + 1), text, color=OVERLAY_SHADOW, scale=1
    )
    frame.draw_string((x, y), text, color=color, scale=1)


def _draw_status_without_detection(frame, status):
    _draw_text(frame, 4, 4, "A4 CALIBRATION")
    _draw_text(frame, 4, 18, status, color=OVERLAY_ERROR)
    _draw_text(
        frame,
        4,
        frame.height() - 28,
        "LONG EDGE -> RIGHT (Y=297MM)",
    )
    _draw_text(
        frame,
        4,
        frame.height() - 14,
        "SHORT EDGE -> DOWN (X=210MM)",
    )


def draw_a4_detection(frame, detection, status, lens=None):
    """Draw the detected A4 outline, named corners, and live quality."""
    corners = detection["corners"]
    contour = detection.get("contour")
    labels = ("TL O", "TR", "BR", "BL")

    # Cyan follows the actual red-mask boundary. Green is the fitted
    # quadrilateral used for calibration. Showing both makes a bad threshold
    # immediately distinguishable from a bad geometric fit.
    if contour:
        stride = max(
            1,
            len(contour) // MAX_OVERLAY_CONTOUR_SEGMENTS,
        )
        sampled = contour[::stride]
        if len(sampled) >= 2:
            for index in range(len(sampled)):
                start = sampled[index]
                end = sampled[(index + 1) % len(sampled)]
                frame.draw_line(
                    (
                        int(start[0]),
                        int(start[1]),
                        int(end[0]),
                        int(end[1]),
                    ),
                    color=(0, 255, 255),
                    thickness=1,
                )

    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        line = (
            int(round(start[0])),
            int(round(start[1])),
            int(round(end[0])),
            int(round(end[1])),
        )
        frame.draw_line(line, color=OVERLAY_SHADOW, thickness=7)
        frame.draw_line(line, color=OVERLAY_FRAME, thickness=4)

    for index, corner in enumerate(corners):
        x = int(round(corner[0]))
        y = int(round(corner[1]))
        frame.draw_circle(
            (x, y, 8), color=OVERLAY_SHADOW, thickness=5
        )
        frame.draw_circle(
            (x, y, 6), color=OVERLAY_CORNER, thickness=3
        )
        label_x = max(0, min(frame.width() - 44, x + 7))
        label_y = max(0, min(frame.height() - 12, y - 12))
        _draw_text(
            frame, label_x, label_y, labels[index], OVERLAY_CORNER
        )

    quality = detection["quality"]
    _draw_text(frame, 4, 4, "A4: %s" % status)
    _draw_text(
        frame,
        4,
        18,
        "EDGE max=%.3fmm rms=%.3fmm"
        % (
            quality["max_edge_residual_mm"],
            quality["rms_edge_residual_mm"],
        ),
    )
    _draw_text(
        frame,
        4,
        32,
        "MARGIN=%.1fpx" % detection["margin_px"],
    )
    if lens is not None:
        _draw_text(
            frame,
            4,
            46,
            "LENS s=%.2f cx=%.1f cy=%.1f"
            % (
                lens["strength"],
                lens["x_corr"],
                lens["y_corr"],
            ),
        )
    _draw_text(
        frame,
        4,
        frame.height() - 14,
        "CYAN=MASK GREEN=FIT | X DOWN Y RIGHT",
    )


def _nearest_contour_indices(contour, corners):
    indices = []
    for corner in corners:
        nearest = min(
            range(len(contour)),
            key=lambda index: geometry.distance(
                contour[index], corner
            ),
        )
        indices.append(nearest)
    indices = sorted(set(indices))
    return indices if len(indices) == 4 else None


def _refined_contour_corners(contour):
    raw_corners = calibration.contour_quad_corners(contour)
    indices = _nearest_contour_indices(contour, raw_corners)
    if indices is None:
        return raw_corners
    refined = geometry.refine_polygon(
        contour,
        indices,
        maximum_shift=MAX_CORNER_REFINEMENT_SHIFT_PX,
    )
    if len(refined) != 4:
        return raw_corners
    refined = calibration.order_quad_corners(refined)
    raw_area = abs(calibration.signed_area(raw_corners))
    refined_area = abs(calibration.signed_area(refined))
    if raw_area <= 0:
        return raw_corners
    area_ratio = refined_area / raw_area
    if area_ratio < 0.88 or area_ratio > 1.12:
        return raw_corners
    return refined


def detect_red_a4(frame):
    """Return the largest valid red A4 detection and its binary contour."""
    try:
        mask = frame.binary(
            [RED_THRESHOLD],
            to_bitmap=True,
            copy=True,
        )
        mask.dilate(1)
        mask.erode(1)
        blobs = mask.find_blobs(
            [(1, 1)],
            x_stride=1,
            y_stride=1,
            area_threshold=MIN_PAPER_AREA_PX,
            pixels_threshold=MIN_PAPER_PIXELS,
            merge=False,
        )
    except Exception as error:
        return None, "red_mask:%s" % str(error)

    if not blobs:
        del mask
        return None, "red_a4_not_found"

    blobs = sorted(blobs, key=_blob_pixels, reverse=True)
    last_reason = "red_region_not_valid_a4"
    for blob in blobs:
        rect = _blob_rect(blob)
        seed = _find_blob_boundary_seed(mask, blob, rect)
        if seed is None:
            last_reason = "red_boundary_seed_not_found"
            continue
        maximum_steps = min(
            MAX_CONTOUR_POINTS,
            max(500, _blob_perimeter(blob) * 4),
        )
        contour, closed = trace_outer_boundary(
            mask, seed, maximum_steps
        )
        if not closed or contour is None or len(contour) < 100:
            last_reason = "red_boundary_trace_not_closed"
            continue

        try:
            corners = _refined_contour_corners(contour)
            usable, reason = calibration.quad_is_usable(
                corners,
                frame.width(),
                frame.height(),
                minimum_margin=PAPER_MARGIN_PX,
            )
            if not usable:
                last_reason = reason
                continue
            horizontal_span = (
                geometry.distance(corners[0], corners[1])
                + geometry.distance(corners[2], corners[3])
            ) * 0.5
            vertical_span = (
                geometry.distance(corners[1], corners[2])
                + geometry.distance(corners[3], corners[0])
            ) * 0.5
            if horizontal_span <= vertical_span * 1.10:
                last_reason = "a4_orientation_not_long_right"
                continue
            quality = calibration.edge_straightness_quality(
                contour, corners
            )
        except (TypeError, ValueError) as error:
            last_reason = "quad_fit:%s" % str(error)
            continue

        detection = {
            "corners": corners,
            "contour": contour,
            "quality": quality,
            "margin_px": calibration.quad_frame_margin(
                corners, frame.width(), frame.height()
            ),
        }
        del mask
        return detection, None

    del mask
    return None, last_reason


def _apply_candidate_lens(frame, lens):
    frame.lens_corr(
        strength=lens["strength"],
        zoom=1.0,
        x_corr=lens["x_corr"],
        y_corr=lens["y_corr"],
    )
    return frame


def _candidate_score(detection):
    quality = detection["quality"]
    return (
        quality["max_edge_residual_mm"]
        + (0.25 * quality["rms_edge_residual_mm"])
    )


def evaluate_lens_candidate(camera, lens, label):
    scores = []
    representative = None
    for frame_index in range(CANDIDATE_FRAMES):
        frame = camera.snapshot()
        _apply_candidate_lens(frame, lens)
        detection, reason = detect_red_a4(frame)
        if detection is None:
            _draw_status_without_detection(
                frame, "%s: %s" % (label, reason)
            )
        else:
            scores.append(_candidate_score(detection))
            representative = detection
            draw_a4_detection(
                frame,
                detection,
                "%s %d/%d"
                % (label, frame_index + 1, CANDIDATE_FRAMES),
                lens=lens,
            )
        camera.flush()
        gc.collect()

    if len(scores) != CANDIDATE_FRAMES:
        return None, representative
    scores.sort()
    return scores[len(scores) // 2], representative


def _float_values(start, stop, step):
    values = []
    value = float(start)
    while value <= stop + (step * 0.25):
        values.append(round(value, 4))
        value += step
    return values


def _search_parameter(camera, best_lens, parameter, values, label):
    best_score = None
    best_candidate = dict(best_lens)
    for index, value in enumerate(values):
        candidate = dict(best_lens)
        candidate[parameter] = float(value)
        score, _ = evaluate_lens_candidate(
            camera,
            candidate,
            "%s %d/%d" % (label, index + 1, len(values)),
        )
        print(
            "CAL_SEARCH %s=%.4f score=%s"
            % (
                parameter,
                value,
                "invalid" if score is None else "%.5f" % score,
            )
        )
        if score is not None and (
            best_score is None or score < best_score
        ):
            best_score = score
            best_candidate = candidate
    if best_score is None:
        raise RuntimeError("no_valid_candidate_for_%s" % parameter)
    return best_candidate, best_score


def search_lens_parameters(camera):
    best = {
        "strength": 1.8,
        "zoom": 1.0,
        "x_corr": 0.0,
        "y_corr": 0.0,
    }

    best, score = _search_parameter(
        camera,
        best,
        "strength",
        _float_values(
            LENS_STRENGTH_MIN,
            LENS_STRENGTH_MAX,
            LENS_STRENGTH_COARSE_STEP,
        ),
        "COARSE STRENGTH",
    )
    offsets = list(
        range(
            -OPTICAL_CENTER_RANGE_PX,
            OPTICAL_CENTER_RANGE_PX + 1,
            OPTICAL_CENTER_COARSE_STEP_PX,
        )
    )
    best, score = _search_parameter(
        camera, best, "x_corr", offsets, "COARSE CENTER-X"
    )
    best, score = _search_parameter(
        camera, best, "y_corr", offsets, "COARSE CENTER-Y"
    )

    best, score = _search_parameter(
        camera,
        best,
        "strength",
        _float_values(
            max(LENS_STRENGTH_MIN, best["strength"] - 0.20),
            min(LENS_STRENGTH_MAX, best["strength"] + 0.20),
            0.05,
        ),
        "FINE STRENGTH",
    )
    best, score = _search_parameter(
        camera,
        best,
        "x_corr",
        range(
            max(
                -OPTICAL_CENTER_RANGE_PX,
                int(round(best["x_corr"])) - 4,
            ),
            min(
                OPTICAL_CENTER_RANGE_PX,
                int(round(best["x_corr"])) + 4,
            )
            + 1,
        ),
        "FINE CENTER-X",
    )
    best, score = _search_parameter(
        camera,
        best,
        "y_corr",
        range(
            max(
                -OPTICAL_CENTER_RANGE_PX,
                int(round(best["y_corr"])) - 4,
            ),
            min(
                OPTICAL_CENTER_RANGE_PX,
                int(round(best["y_corr"])) + 4,
            )
            + 1,
        ),
        "FINE CENTER-Y",
    )
    return best, score


def _wait_for_stable_paper(camera):
    stable = 0
    print("CALIBRATION_WAITING_FOR_RED_A4")
    while stable < STABLE_DETECTIONS:
        frame = camera.snapshot()
        detection, reason = detect_red_a4(frame)
        if detection is None:
            stable = 0
            _draw_status_without_detection(
                frame, "PLACE RED A4: %s" % reason
            )
        else:
            stable += 1
            draw_a4_detection(
                frame,
                detection,
                "FOUND - HOLD STILL %d/%d"
                % (stable, STABLE_DETECTIONS),
            )
        camera.flush()
        gc.collect()
    return detection


def _lock_camera(camera):
    gain_db = camera.gain_db()
    exposure_us = camera.exposure_us()
    rgb_gain_db = camera.rgb_gain_db()
    camera.auto_gain(False, gain_db=gain_db)
    camera.auto_exposure(False, exposure_us=exposure_us)
    camera.auto_whitebal(False, rgb_gain_db=rgb_gain_db)
    camera.snapshot(time=250)
    print(
        "CAL_CAMERA_LOCKED gain=%.2fdB exposure=%dus rgb=%s"
        % (gain_db, exposure_us, str(rgb_gain_db))
    )
    return {
        "gain_db": round(gain_db, 4),
        "exposure_us": int(exposure_us),
        "rgb_gain_db": [
            round(float(value), 4) for value in rgb_gain_db
        ],
    }


def collect_final_validation(camera, lens):
    corner_samples = []
    frame_qualities = []
    minimum_margin = None
    for index in range(FINAL_VALIDATION_FRAMES):
        frame = camera.snapshot()
        _apply_candidate_lens(frame, lens)
        detection, reason = detect_red_a4(frame)
        if detection is None:
            _draw_status_without_detection(
                frame, "VALIDATION LOST: %s" % reason
            )
            camera.flush()
            raise RuntimeError("validation_lost_red_a4:%s" % reason)

        corner_samples.append(detection["corners"])
        frame_qualities.append(detection["quality"])
        if minimum_margin is None:
            minimum_margin = detection["margin_px"]
        else:
            minimum_margin = min(
                minimum_margin, detection["margin_px"]
            )
        draw_a4_detection(
            frame,
            detection,
            "VALIDATE %d/%d" % (
                index + 1,
                FINAL_VALIDATION_FRAMES,
            ),
            lens=lens,
        )
        camera.flush()
        if (index + 1) % 10 == 0:
            print(
                "CAL_VALIDATION %d/%d edge_max=%.4fmm"
                % (
                    index + 1,
                    FINAL_VALIDATION_FRAMES,
                    detection["quality"]["max_edge_residual_mm"],
                )
            )
            gc.collect()

    error = calibration.estimate_model_error_mm(
        frame_qualities,
        corner_samples,
        output_width=camera.width(),
        output_height=camera.height(),
    )
    error["sample_count"] = FINAL_VALIDATION_FRAMES
    error["minimum_frame_margin_px"] = round(minimum_margin, 3)
    error["median_edge_rms_mm"] = round(
        calibration._median(
            [
                quality["rms_edge_residual_mm"]
                for quality in frame_qualities
            ]
        ),
        4,
    )
    return corner_samples, error


def _build_config(camera, lens, lock_settings, search_score, quality):
    signature = calibration.camera_signature(camera)
    signature["calibration_pixformat"] = "RGB565"
    signature["runtime_pixformat"] = "GRAYSCALE"
    signature["camera_height_mm_nominal"] = 295
    return {
        "version": calibration.CALIBRATION_VERSION,
        "mode": "lens_only_dynamic_a4",
        "camera": signature,
        "a4": {
            "x_mm": calibration.A4_X_MM,
            "y_mm": calibration.A4_Y_MM,
            "orientation": calibration.ORIENTATION_DESCRIPTION,
            "origin": calibration.ORIGIN_DESCRIPTION,
            "pose_saved": False,
        },
        "lens": {
            "strength": round(float(lens["strength"]), 4),
            "zoom": 1.0,
            "x_corr": round(float(lens["x_corr"]), 4),
            "y_corr": round(float(lens["y_corr"]), 4),
        },
        "quality": {
            "lens_edge_residual_p95_mm": quality[
                "edge_residual_p95_mm"
            ],
            "corner_repeatability_mm": quality[
                "corner_standard_error_mm"
            ],
            "median_edge_rms_mm": quality[
                "median_edge_rms_mm"
            ],
            "minimum_frame_margin_px": quality[
                "minimum_frame_margin_px"
            ],
            "sample_count": quality["sample_count"],
            "search_score": round(float(search_score), 5),
            "paper_pose_saved": False,
            "criterion": "lens_only_live_a4_pose",
        },
        "capture_settings": lock_settings,
        "runtime": {
            "paper_localization": "per_frame_grayscale",
            "dark_threshold": 165,
        },
    }


def _draw_rectified_axes(frame, quality):
    width = frame.width()
    height = frame.height()
    frame.draw_rectangle(
        (2, 2, width - 4, height - 4),
        color=OVERLAY_FRAME,
        thickness=3,
    )
    frame.draw_line((12, 12, 12, 92), color=(255, 255, 0), thickness=4)
    frame.draw_line((12, 92, 7, 82), color=(255, 255, 0), thickness=4)
    frame.draw_line((12, 92, 17, 82), color=(255, 255, 0), thickness=4)
    frame.draw_line((12, 12, 122, 12), color=(0, 255, 255), thickness=4)
    frame.draw_line((122, 12, 112, 7), color=(0, 255, 255), thickness=4)
    frame.draw_line((122, 12, 112, 17), color=(0, 255, 255), thickness=4)
    _draw_text(frame, 18, 78, "X 210MM", (255, 255, 0))
    _draw_text(frame, 80, 20, "Y 297MM", (0, 255, 255))
    _draw_text(
        frame,
        4,
        height - 14,
        "CAL OK estimated max=%.3fmm"
        % quality["estimated_max_error_mm"],
        OVERLAY_FRAME,
    )


def _success_preview(camera, config):
    clock = time.clock()
    frame_count = 0
    print("CALIBRATION_PREVIEW_RECTIFIED")
    try:
        while True:
            clock.tick()
            frame = camera.snapshot()
            calibration.correct_lens(frame, config)
            detection, reason = detect_red_a4(frame)
            if detection is None:
                _draw_status_without_detection(
                    frame, "LIVE A4 LOST: %s" % reason
                )
                camera.flush()
                continue
            frame.rotation_corr(
                corners=[
                    (int(round(point[0])), int(round(point[1])))
                    for point in detection["corners"]
                ]
            )
            _draw_rectified_axes(
                frame,
                {"estimated_max_error_mm": 0.0},
            )
            camera.flush()
            frame_count += 1
            if frame_count >= 30:
                print("RECTIFIED_FPS %.2f" % clock.fps())
                frame_count = 0
    except KeyboardInterrupt:
        camera.flush()
        print("A4_CALIBRATION_STOPPED")


def main():
    camera = csi.CSI()
    camera.reset()
    camera.pixformat(csi.RGB565)
    camera.framesize(csi.VGA)
    camera.hmirror(False)
    camera.vflip(False)
    camera.transpose(False)
    camera.framebuffers(FRAME_BUFFER_COUNT)
    camera.snapshot(time=CAMERA_WARMUP_MS)

    print("A4_CALIBRATION_READY")
    print(
        "A4_ORIENTATION=LONG_RIGHT_SHORT_DOWN X=DOWN_210MM Y=RIGHT_297MM"
    )
    _wait_for_stable_paper(camera)
    lock_settings = _lock_camera(camera)

    print("CALIBRATION_LENS_SEARCH_START")
    lens, search_score = search_lens_parameters(camera)
    print(
        "CALIBRATION_LENS_BEST strength=%.4f x_corr=%.2f y_corr=%.2f score=%.5f"
        % (
            lens["strength"],
            lens["x_corr"],
            lens["y_corr"],
            search_score,
        )
    )

    _, quality = collect_final_validation(camera, lens)
    print("CALIBRATION_QUALITY %s" % str(quality))
    config = _build_config(
        camera, lens, lock_settings, search_score, quality
    )
    calibration.save_calibration(config)
    print(
        "LENS_CALIBRATION_SAVED path=%s paper_pose_saved=false "
        "live_model_estimate=%.4fmm"
        % (
            calibration.CONFIG_PATH,
            quality["estimated_max_error_mm"],
        )
    )
    _success_preview(camera, config)


if __name__ == "__main__":
    main()
