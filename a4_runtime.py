"""Dynamic grayscale A4 localization for the fixed-lens OpenMV camera.

Only lens distortion is persistent. The red A4 pose is detected again from
every grayscale frame so a small lateral camera movement does not move the
physical origin or axes.
"""

import calibration_geometry as calibration
import polygon_geometry as geometry


DARK_A4_THRESHOLD = 165
MIN_PAPER_AREA_PX = 42000
MIN_PAPER_PIXELS = 28000
PAPER_MARGIN_PX = 0
MAX_CONTOUR_POINTS = 9000
MAX_CORNER_REFINEMENT_SHIFT_PX = 14.0
POSE_SMOOTHING_ALPHA = 0.35
MAX_STALE_POSE_FRAMES = 30

_last_corners = None
_stale_pose_frames = 0

_NEIGHBOURS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
)


def _value_or_call(obj, name):
    value = getattr(obj, name)
    return value() if callable(value) else value


def _pixel_value(image_object, x, y):
    point = (int(x), int(y))
    try:
        value = image_object.get_pixel(point)
    except TypeError:
        value = image_object.get_pixel(point[0], point[1])
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _foreground(mask, x, y):
    value = _pixel_value(mask, x, y)
    return value is not None and value != 0


def _blob_rect(blob):
    return tuple(int(value) for value in _value_or_call(blob, "rect"))


def _blob_pixels(blob):
    return int(_value_or_call(blob, "pixels"))


def _blob_perimeter(blob):
    return int(_value_or_call(blob, "perimeter"))


def _find_nearby_foreground(mask, point, rect):
    center_x = int(point[0])
    center_y = int(point[1])
    rect_x, rect_y, rect_width, rect_height = rect
    x_min = max(0, rect_x)
    y_min = max(0, rect_y)
    x_max = min(mask.width() - 1, rect_x + rect_width - 1)
    y_max = min(mask.height() - 1, rect_y + rect_height - 1)
    for radius in range(max(rect_width, rect_height) + 1):
        left = max(x_min, center_x - radius)
        right = min(x_max, center_x + radius)
        top = max(y_min, center_y - radius)
        bottom = min(y_max, center_y + radius)
        for x in range(left, right + 1):
            if _foreground(mask, x, top):
                return x, top
            if bottom != top and _foreground(mask, x, bottom):
                return x, bottom
        for y in range(top + 1, bottom):
            if _foreground(mask, left, y):
                return left, y
            if right != left and _foreground(mask, right, y):
                return right, y
    return None


def _boundary_seed(mask, blob, rect):
    try:
        center = (
            int(_value_or_call(blob, "cx")),
            int(_value_or_call(blob, "cy")),
        )
    except (AttributeError, TypeError):
        center = (
            rect[0] + (rect[2] // 2),
            rect[1] + (rect[3] // 2),
        )
    anchor = center
    if not _foreground(mask, anchor[0], anchor[1]):
        anchor = _find_nearby_foreground(mask, center, rect)
    if anchor is None:
        return None
    x, y = anchor
    left = max(0, rect[0])
    while x > left and _foreground(mask, x - 1, y):
        x -= 1
    return x, y


def _nearby_boundary(mask, point, radius=2):
    center_x, center_y = int(point[0]), int(point[1])
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            x = center_x + offset_x
            y = center_y + offset_y
            if not _foreground(mask, x, y):
                continue
            for dx, dy in _NEIGHBOURS:
                if not _foreground(mask, x + dx, y + dy):
                    return x, y
    return None


def _trace_boundary(mask, start, maximum_steps):
    if start is None:
        return None, False
    if (
        not _foreground(mask, start[0], start[1])
        or _foreground(mask, start[0] - 1, start[1])
    ):
        start = _nearby_boundary(mask, start)
    if start is None:
        return None, False

    current = start
    backtrack = (start[0] - 1, start[1])
    first_next = None
    contour = [start]
    for _ in range(maximum_steps):
        relative = (
            backtrack[0] - current[0],
            backtrack[1] - current[1],
        )
        try:
            backtrack_index = _NEIGHBOURS.index(relative)
        except ValueError:
            backtrack_index = 7

        next_point = None
        next_index = None
        for step in range(1, 9):
            index = (backtrack_index + step) % 8
            dx, dy = _NEIGHBOURS[index]
            candidate = (current[0] + dx, current[1] + dy)
            if _foreground(mask, candidate[0], candidate[1]):
                next_point = candidate
                next_index = index
                break
        if next_point is None:
            return contour, False
        if first_next is None:
            first_next = next_point
        elif current == start and next_point == first_next:
            if len(contour) > 1 and contour[-1] == start:
                contour.pop()
            return contour, len(contour) >= 3

        previous_dx, previous_dy = _NEIGHBOURS[
            (next_index - 1) % 8
        ]
        backtrack = (
            current[0] + previous_dx,
            current[1] + previous_dy,
        )
        current = next_point
        contour.append(current)
    return contour, False


def _refine_corners(contour):
    raw = calibration.contour_quad_corners(contour)
    indices = []
    for corner in raw:
        indices.append(
            min(
                range(len(contour)),
                key=lambda index: geometry.distance(
                    contour[index], corner
                ),
            )
        )
    indices = sorted(set(indices))
    if len(indices) != 4:
        return raw
    refined = geometry.refine_polygon(
        contour,
        indices,
        maximum_shift=MAX_CORNER_REFINEMENT_SHIFT_PX,
    )
    if len(refined) != 4:
        return raw
    refined = calibration.order_quad_corners(refined)
    raw_area = abs(calibration.signed_area(raw))
    refined_area = abs(calibration.signed_area(refined))
    if raw_area <= 0:
        return raw
    ratio = refined_area / raw_area
    return refined if 0.88 <= ratio <= 1.12 else raw


def detect_a4(frame):
    """Return current grayscale A4 corners after lens correction."""
    try:
        mask = frame.binary(
            [(0, DARK_A4_THRESHOLD)],
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
        return None, "a4_mask:%s" % str(error)

    if not blobs:
        del mask
        return None, "a4_dark_region_not_found"

    last_reason = "a4_region_not_valid"
    for blob in sorted(blobs, key=_blob_pixels, reverse=True):
        rect = _blob_rect(blob)
        seed = _boundary_seed(mask, blob, rect)
        contour, closed = _trace_boundary(
            mask,
            seed,
            min(
                MAX_CONTOUR_POINTS,
                max(500, _blob_perimeter(blob) * 4),
            ),
        )
        if not closed or contour is None or len(contour) < 100:
            last_reason = "a4_boundary_not_closed"
            continue
        try:
            corners = _refine_corners(contour)
            usable, reason = calibration.quad_is_usable(
                corners,
                frame.width(),
                frame.height(),
                minimum_margin=PAPER_MARGIN_PX,
            )
            if not usable:
                last_reason = reason
                continue
            horizontal = (
                geometry.distance(corners[0], corners[1])
                + geometry.distance(corners[2], corners[3])
            ) * 0.5
            vertical = (
                geometry.distance(corners[1], corners[2])
                + geometry.distance(corners[3], corners[0])
            ) * 0.5
            if horizontal <= vertical * 1.10:
                last_reason = "a4_orientation_not_long_right"
                continue
        except (TypeError, ValueError) as error:
            last_reason = "a4_quad:%s" % str(error)
            continue

        result = {
            "corners": corners,
            "margin_px": calibration.quad_frame_margin(
                corners, frame.width(), frame.height()
            ),
        }
        del mask
        return result, None

    del mask
    return None, last_reason


def reset_pose_tracker():
    global _last_corners, _stale_pose_frames
    _last_corners = None
    _stale_pose_frames = 0


def _tracked_corners(detection):
    global _last_corners, _stale_pose_frames
    if detection is None:
        if (
            _last_corners is None
            or _stale_pose_frames >= MAX_STALE_POSE_FRAMES
        ):
            return None, False
        _stale_pose_frames += 1
        return _last_corners, True

    live = calibration.order_quad_corners(detection["corners"])
    if _last_corners is None:
        tracked = live
    else:
        alpha = POSE_SMOOTHING_ALPHA
        tracked = []
        for old, new in zip(_last_corners, live):
            tracked.append(
                (
                    old[0] + ((new[0] - old[0]) * alpha),
                    old[1] + ((new[1] - old[1]) * alpha),
                )
            )
    _last_corners = tracked
    _stale_pose_frames = 0
    return tracked, False


def rectify_to_a4(frame, config):
    """Lens-correct, locate the current A4, then rectify to its live pose."""
    calibration.correct_lens(frame, config)
    detection, reason = detect_a4(frame)
    corners, used_cached_pose = _tracked_corners(detection)
    if corners is None:
        raise RuntimeError(reason)
    frame.rotation_corr(
        corners=[
            (int(round(point[0])), int(round(point[1])))
            for point in corners
        ]
    )
    if detection is None:
        return {
            "corners": corners,
            "pose_source": "recent_valid_frame",
            "stale_frames": _stale_pose_frames,
        }
    detection["corners"] = corners
    detection["pose_source"] = (
        "recent_valid_frame" if used_cached_pose else "live_frame"
    )
    detection["stale_frames"] = 0
    return detection
