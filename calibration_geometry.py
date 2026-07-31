"""A4-plane calibration helpers shared by OpenMV and host-side tests.

The physical coordinate system is intentionally different from normal image
coordinates:

* the red A4 sheet's top-left corner is (X=0, Y=0);
* X increases down the 210 mm short edge;
* Y increases right along the 297 mm long edge.

The module avoids CPython-only dependencies so it can run unchanged under
OpenMV MicroPython.
"""

import math
import os

try:
    import json
except ImportError:
    import ujson as json


CALIBRATION_VERSION = 2
# OpenMV exposes the USB flash volume as the script working directory. A
# relative path works both there and in host tests; an absolute "/" path can
# address the MicroPython VFS root instead of the mounted flash volume.
CONFIG_PATH = "a4_calibration.json"

A4_X_MM = 210.0
A4_Y_MM = 297.0
EXPECTED_FRAME_WIDTH = 640
EXPECTED_FRAME_HEIGHT = 480
MAX_MODEL_ERROR_MM = 1.0

ORIGIN_DESCRIPTION = "red_a4_top_left"
X_DIRECTION = "down"
Y_DIRECTION = "right"
ORIENTATION_DESCRIPTION = "long_edge_right_short_edge_down"

_EPSILON = 1e-9


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


def _value_or_call(obj, name):
    value = getattr(obj, name)
    return value() if callable(value) else value


def _mean(values):
    return (sum(values) / len(values)) if values else 0.0


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return 0.0
    midpoint = count // 2
    if count % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) * 0.5


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = int(round((len(ordered) - 1) * fraction))
    position = max(0, min(len(ordered) - 1, position))
    return float(ordered[position])


def distance(point_a, point_b):
    dx = float(point_b[0]) - float(point_a[0])
    dy = float(point_b[1]) - float(point_a[1])
    return math.sqrt((dx * dx) + (dy * dy))


def signed_area(points):
    total = 0.0
    for index in range(len(points)):
        first = points[index]
        second = points[(index + 1) % len(points)]
        total += (first[0] * second[1]) - (first[1] * second[0])
    return total * 0.5


def order_quad_corners(points):
    """Return TL, TR, BR, BL for one convex image-coordinate quadrilateral."""
    if len(points) != 4:
        raise ValueError("expected exactly four corners")

    converted = [
        (float(point[0]), float(point[1])) for point in points
    ]
    center_x = _mean([point[0] for point in converted])
    center_y = _mean([point[1] for point in converted])
    ordered = sorted(
        converted,
        key=lambda point: math.atan2(
            point[1] - center_y, point[0] - center_x
        ),
    )

    # In an image whose y coordinate increases downwards, increasing atan2
    # order follows the visible boundary clockwise.
    if signed_area(ordered) < 0:
        ordered.reverse()

    top_left_index = min(
        range(4),
        key=lambda index: (
            ordered[index][0] + ordered[index][1],
            ordered[index][1],
            ordered[index][0],
        ),
    )
    ordered = ordered[top_left_index:] + ordered[:top_left_index]

    # The first neighbour must be the top/right corner rather than bottom/left.
    if ordered[1][0] < ordered[-1][0]:
        ordered = [ordered[0], ordered[-1], ordered[-2], ordered[-3]]
    return ordered


def contour_quad_corners(contour):
    """Estimate TL/TR/BR/BL directly from a closed paper contour.

    OpenMV ``Blob.corners`` describes the blob's bounding rectangle, not the
    four perspective corners of the segmented object.  Using normalized
    diagonal extrema keeps this independent of the A4 aspect ratio and makes
    the initial corners land on the real red-mask boundary.
    """
    if contour is None or len(contour) < 4:
        raise ValueError("contour requires at least four points")

    points = [
        (float(point[0]), float(point[1])) for point in contour
    ]
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    if width < _EPSILON or height < _EPSILON:
        raise ValueError("degenerate contour bounds")

    def normalized(point):
        return (
            (point[0] - minimum_x) / width,
            (point[1] - minimum_y) / height,
        )

    top_left = min(
        points,
        key=lambda point: sum(normalized(point)),
    )
    top_right = min(
        points,
        key=lambda point: (
            (1.0 - normalized(point)[0])
            + normalized(point)[1]
        ),
    )
    bottom_right = min(
        points,
        key=lambda point: (
            (1.0 - normalized(point)[0])
            + (1.0 - normalized(point)[1])
        ),
    )
    bottom_left = min(
        points,
        key=lambda point: (
            normalized(point)[0]
            + (1.0 - normalized(point)[1])
        ),
    )
    corners = [top_left, top_right, bottom_right, bottom_left]
    if len(set(corners)) != 4:
        raise ValueError("contour extrema did not produce four corners")
    return order_quad_corners(corners)


def quad_frame_margin(corners, width, height):
    ordered = order_quad_corners(corners)
    margins = []
    for x, y in ordered:
        margins.extend((x, y, (width - 1) - x, (height - 1) - y))
    return min(margins)


def quad_is_usable(corners, width, height, minimum_margin=12.0):
    try:
        ordered = order_quad_corners(corners)
    except (TypeError, ValueError):
        return False, "not_a_quad"

    area = abs(signed_area(ordered))
    if area < width * height * 0.12:
        return False, "paper_too_small"
    if quad_frame_margin(ordered, width, height) < minimum_margin:
        return False, "paper_clipped_or_margin_too_small"

    edge_lengths = [
        distance(ordered[index], ordered[(index + 1) % 4])
        for index in range(4)
    ]
    if min(edge_lengths) < min(width, height) * 0.20:
        return False, "paper_edge_too_short"
    return True, None


def _point_line_distance(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = math.sqrt((dx * dx) + (dy * dy))
    if denominator < _EPSILON:
        return distance(point, start)
    numerator = abs(
        (dy * point[0])
        - (dx * point[1])
        + (end[0] * start[1])
        - (end[1] * start[0])
    )
    return numerator / denominator


def _robust_residual_summary(values):
    if not values:
        return 0.0, 0.0, 0
    center = _median(values)
    mad = _median([abs(value - center) for value in values])
    cutoff = center + max(0.20, 3.0 * 1.4826 * mad)
    inliers = [value for value in values if value <= cutoff]
    if not inliers:
        inliers = list(values)
    rms = math.sqrt(_mean([value * value for value in inliers]))
    return max(inliers), rms, len(inliers)


def edge_straightness_quality(contour, corners):
    """Measure A4 edge curvature after a candidate lens correction.

    Perspective preserves straight lines, so a good radial correction makes
    the four paper borders straight even before the projective rectification.
    """
    ordered = order_quad_corners(corners)
    side_lengths = [
        distance(ordered[index], ordered[(index + 1) % 4])
        for index in range(4)
    ]
    vertical_span_px = max(
        _EPSILON, (side_lengths[1] + side_lengths[3]) * 0.5
    )
    horizontal_span_px = max(
        _EPSILON, (side_lengths[0] + side_lengths[2]) * 0.5
    )
    x_scale = A4_X_MM / vertical_span_px
    y_scale = A4_Y_MM / horizontal_span_px
    normal_scales = (x_scale, y_scale, x_scale, y_scale)

    residuals_by_side = [[], [], [], []]
    for raw_point in contour:
        point = (float(raw_point[0]), float(raw_point[1]))
        distances = [
            _point_line_distance(
                point, ordered[index], ordered[(index + 1) % 4]
            )
            for index in range(4)
        ]
        side_index = min(range(4), key=lambda index: distances[index])
        residuals_by_side[side_index].append(
            distances[side_index] * normal_scales[side_index]
        )

    maximums = []
    squared_total = 0.0
    inlier_total = 0
    side_inliers = []
    for residuals in residuals_by_side:
        maximum, rms, count = _robust_residual_summary(residuals)
        maximums.append(maximum)
        squared_total += (rms * rms) * count
        inlier_total += count
        side_inliers.append(count)

    return {
        "max_edge_residual_mm": round(max(maximums), 4),
        "rms_edge_residual_mm": round(
            math.sqrt(squared_total / max(1, inlier_total)), 4
        ),
        "x_mm_per_pixel": x_scale,
        "y_mm_per_pixel": y_scale,
        "side_inlier_counts": side_inliers,
    }


def average_corners(corner_samples):
    if not corner_samples:
        raise ValueError("corner samples are required")
    ordered_samples = [
        order_quad_corners(sample) for sample in corner_samples
    ]
    result = []
    for corner_index in range(4):
        result.append(
            (
                _mean(
                    [
                        sample[corner_index][0]
                        for sample in ordered_samples
                    ]
                ),
                _mean(
                    [
                        sample[corner_index][1]
                        for sample in ordered_samples
                    ]
                ),
            )
        )
    return result


def corner_standard_error_mm(corner_samples, mean_corners=None):
    """Return the largest corner mean-position standard error in millimetres."""
    if not corner_samples:
        return 0.0
    ordered_samples = [
        order_quad_corners(sample) for sample in corner_samples
    ]
    if mean_corners is None:
        mean_corners = average_corners(ordered_samples)
    mean_corners = order_quad_corners(mean_corners)

    side_lengths = [
        distance(mean_corners[index], mean_corners[(index + 1) % 4])
        for index in range(4)
    ]
    x_scale = A4_X_MM / max(
        _EPSILON, (side_lengths[1] + side_lengths[3]) * 0.5
    )
    y_scale = A4_Y_MM / max(
        _EPSILON, (side_lengths[0] + side_lengths[2]) * 0.5
    )

    maximum_standard_error = 0.0
    sample_count = len(ordered_samples)
    for corner_index in range(4):
        squared_distances = []
        mean_x, mean_y = mean_corners[corner_index]
        for sample in ordered_samples:
            dx_mm = (sample[corner_index][0] - mean_x) * y_scale
            dy_mm = (sample[corner_index][1] - mean_y) * x_scale
            squared_distances.append(
                (dx_mm * dx_mm) + (dy_mm * dy_mm)
            )
        sigma = math.sqrt(_mean(squared_distances))
        standard_error = sigma / math.sqrt(sample_count)
        maximum_standard_error = max(
            maximum_standard_error, standard_error
        )
    return maximum_standard_error


def estimate_model_error_mm(
    frame_qualities,
    corner_samples,
    output_width=EXPECTED_FRAME_WIDTH,
    output_height=EXPECTED_FRAME_HEIGHT,
):
    edge_maxima = [
        float(quality["max_edge_residual_mm"])
        for quality in frame_qualities
    ]
    mean_corners = average_corners(corner_samples)
    corner_error = corner_standard_error_mm(
        corner_samples, mean_corners
    )
    half_pixel_error = 0.5 * max(
        A4_X_MM / max(1, output_height - 1),
        A4_Y_MM / max(1, output_width - 1),
    )
    edge_error = _percentile(edge_maxima, 0.95)
    estimated = edge_error + (3.0 * corner_error) + half_pixel_error
    return {
        "estimated_max_error_mm": round(estimated, 4),
        "edge_residual_p95_mm": round(edge_error, 4),
        "corner_standard_error_mm": round(corner_error, 4),
        "half_pixel_error_mm": round(half_pixel_error, 4),
        "mean_corners_px": [
            [round(point[0], 4), round(point[1], 4)]
            for point in mean_corners
        ],
    }


def pixel_to_xy_mm(point, width, height, clamp=False):
    """Convert rectified image (u, v) to physical (X-down, Y-right) mm."""
    u = float(point[0])
    v = float(point[1])
    if clamp:
        u = max(0.0, min(width - 1.0, u))
        v = max(0.0, min(height - 1.0, v))
    x_mm = v * A4_X_MM / max(1.0, height - 1.0)
    y_mm = u * A4_Y_MM / max(1.0, width - 1.0)
    return x_mm, y_mm


def metric_polygon_measurements(vertices_px, width, height):
    vertices_xy = [
        pixel_to_xy_mm(point, width, height) for point in vertices_px
    ]
    edge_lengths = [
        distance(
            vertices_xy[index],
            vertices_xy[(index + 1) % len(vertices_xy)],
        )
        for index in range(len(vertices_xy))
    ]

    # Polygon centroid in metric X/Y coordinates.
    area_twice = 0.0
    x_total = 0.0
    y_total = 0.0
    for index in range(len(vertices_xy)):
        x1, y1 = vertices_xy[index]
        x2, y2 = vertices_xy[(index + 1) % len(vertices_xy)]
        cross = (x1 * y2) - (x2 * y1)
        area_twice += cross
        x_total += (x1 + x2) * cross
        y_total += (y1 + y2) * cross
    if abs(area_twice) < _EPSILON:
        centroid_xy = (
            _mean([point[0] for point in vertices_xy]),
            _mean([point[1] for point in vertices_xy]),
        )
    else:
        scale = 1.0 / (3.0 * area_twice)
        centroid_xy = (x_total * scale, y_total * scale)

    return {
        "vertices_xy_mm": [
            [round(point[0], 3), round(point[1], 3)]
            for point in vertices_xy
        ],
        "centroid_xy_mm": [
            round(centroid_xy[0], 3),
            round(centroid_xy[1], 3),
        ],
        "edge_lengths_mm": [
            round(length, 3) for length in edge_lengths
        ],
    }


def physical_coordinate_system(config=None):
    result = {
        "origin": ORIGIN_DESCRIPTION,
        "x_direction": X_DIRECTION,
        "y_direction": Y_DIRECTION,
        "unit": "mm",
        "x_range_mm": [0.0, A4_X_MM],
        "y_range_mm": [0.0, A4_Y_MM],
        "a4_orientation": ORIENTATION_DESCRIPTION,
    }
    if config is not None:
        result["calibration_version"] = config.get("version")
        estimated_error = config.get("quality", {}).get(
            "estimated_max_error_mm"
        )
        if estimated_error is not None:
            result["estimated_max_error_mm"] = estimated_error
        result["paper_pose"] = "detected_each_frame"
    return result


def camera_signature(camera):
    return {
        "sensor_id": int(_value_or_call(camera, "cid")),
        "width": int(camera.width()),
        "height": int(camera.height()),
        "hmirror": bool(_value_or_call(camera, "hmirror")),
        "vflip": bool(_value_or_call(camera, "vflip")),
        "transpose": bool(_value_or_call(camera, "transpose")),
    }


def validate_calibration(config, signature=None):
    if not isinstance(config, dict):
        return False, "config_not_object"
    if config.get("version") != CALIBRATION_VERSION:
        return False, "unsupported_calibration_version"

    camera = config.get("camera")
    lens = config.get("lens")
    paper = config.get("a4")
    quality = config.get("quality")
    if not all(
        isinstance(value, dict)
        for value in (camera, lens, paper, quality)
    ):
        return False, "missing_calibration_sections"
    if config.get("mode") != "lens_only_dynamic_a4":
        return False, "unexpected_calibration_mode"
    if "rectification" in config:
        return False, "saved_paper_pose_not_allowed"

    if (
        camera.get("width") != EXPECTED_FRAME_WIDTH
        or camera.get("height") != EXPECTED_FRAME_HEIGHT
    ):
        return False, "unexpected_calibration_resolution"
    if (
        float(paper.get("x_mm", -1)) != A4_X_MM
        or float(paper.get("y_mm", -1)) != A4_Y_MM
        or paper.get("orientation") != ORIENTATION_DESCRIPTION
        or paper.get("pose_saved") is not False
    ):
        return False, "unexpected_a4_geometry"

    lens_values = (
        lens.get("strength"),
        lens.get("zoom"),
        lens.get("x_corr"),
        lens.get("y_corr"),
    )
    if not all(_finite(value) for value in lens_values):
        return False, "invalid_lens_parameters"
    if not (0.1 <= float(lens.get("strength")) <= 5.0):
        return False, "lens_strength_out_of_range"
    if not (0.5 <= float(lens.get("zoom")) <= 2.0):
        return False, "lens_zoom_out_of_range"

    if quality.get("paper_pose_saved") is not False:
        return False, "quality_pose_flag_invalid"

    if signature is not None:
        for key in (
            "sensor_id",
            "width",
            "height",
            "hmirror",
            "vflip",
            "transpose",
        ):
            # A manually provisioned lens-only config may intentionally leave
            # the firmware-specific sensor identifier unspecified. Geometry
            # and orientation fields remain mandatory and strictly checked.
            if key == "sensor_id" and camera.get(key) is None:
                continue
            if camera.get(key) != signature.get(key):
                return False, "camera_signature_mismatch_%s" % key
    return True, None


def load_calibration(path=CONFIG_PATH, signature=None):
    try:
        with open(path, "r") as config_file:
            config = json.loads(config_file.read())
    except OSError:
        return None, "calibration_file_missing"
    except (TypeError, ValueError):
        return None, "calibration_json_invalid"

    valid, reason = validate_calibration(config, signature=signature)
    return (config, None) if valid else (None, reason)


def _exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def save_calibration(config, path=CONFIG_PATH):
    """Validate, write, verify, and retain one last-known-good backup."""
    valid, reason = validate_calibration(config)
    if not valid:
        raise ValueError(reason)

    temporary_path = path + ".tmp"
    backup_path = path + ".bak"
    serialized = json.dumps(config)
    with open(temporary_path, "w") as config_file:
        config_file.write(serialized)

    with open(temporary_path, "r") as verify_file:
        verified = json.loads(verify_file.read())
    valid, reason = validate_calibration(verified)
    if not valid:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        raise ValueError("written_config_invalid:%s" % reason)

    old_moved = False
    try:
        if _exists(backup_path):
            os.remove(backup_path)
        if _exists(path):
            os.rename(path, backup_path)
            old_moved = True
        os.rename(temporary_path, path)
    except Exception:
        if old_moved and not _exists(path) and _exists(backup_path):
            os.rename(backup_path, path)
        raise
    return path


def correct_lens(frame, config):
    lens = config["lens"]
    frame.lens_corr(
        strength=float(lens["strength"]),
        zoom=float(lens["zoom"]),
        x_corr=float(lens["x_corr"]),
        y_corr=float(lens["y_corr"]),
    )
    return frame


def rectify_frame(frame, config, corners=None):
    """Apply fixed lens correction and a caller-supplied live A4 pose."""
    correct_lens(frame, config)
    if corners is None:
        raise ValueError("live_a4_corners_required")
    ordered = order_quad_corners(corners)
    frame.rotation_corr(
        corners=[
            (int(round(point[0])), int(round(point[1])))
            for point in ordered
        ]
    )
    return frame
