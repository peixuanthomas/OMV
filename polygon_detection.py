"""Detect up to four three-to-five-sided polygons in fixed-camera images.

Target runtime: OpenMV firmware 4.8+/5.x using the ``csi.CSI()`` camera API.
Camera setup is shared with ``openmv_edge_detection.py``. The fixed-camera
path uses threshold segmentation and robust line fitting; optional Canny
validation can be enabled for difficult scenes. The IDE frame buffer shows
only accepted polygon edges over the original grayscale stream, with
pixel-only vertex and edge measurements.
"""

import gc
import math
import time

try:
    import json
except ImportError:
    import ujson as json

import polygon_geometry as geometry
from openmv_edge_detection import (
    EDGE_THRESHOLD,
    init_camera,
    process_image as detect_edges_in_place,
)


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

# The fixed fixture uses light polygon pieces over a darker work surface.
# Keeping this explicit prevents dark objects at the frame border from
# reversing the segmentation polarity.
FOREGROUND_POLARITY = "bright"

# Fixed for the current camera height and lighting: the white piece is around
# 200+ gray levels while the work surface is roughly 150-160. Keeping a
# 20-level margin preserves the dimmer pointed end of the white piece.
FIXED_FOREGROUND_THRESHOLD = 180
MINIMUM_EDGE_PX = 20.0
# The shortest supported piece edge is about 20 px in the fixed fixture.
# Perspective, threshold stair-steps and line intersections can measure it at
# 16-19 px, so keep the physical target explicit while validating against a
# small measurement tolerance.
EDGE_LENGTH_TOLERANCE_PX = 4.0
MINIMUM_VALIDATED_EDGE_PX = (
    MINIMUM_EDGE_PX - EDGE_LENGTH_TOLERANCE_PX
)
LOCK_CAMERA_SETTINGS = True


# ---------------------------------------------------------------------------
# Segmentation and polygon configuration
# ---------------------------------------------------------------------------

MAX_POLYGONS = 4
MAX_CANDIDATE_BLOBS = 12
MIN_BLOB_PIXELS = 480
MIN_BLOB_BOUNDING_AREA = 720
MIN_POLYGON_AREA_PX = 600.0
FRAME_BORDER_MARGIN_PX = 4

OTSU_MARGIN = 5
MIN_AUTO_THRESHOLD = 20
MAX_AUTO_THRESHOLD = 235
MEDIAN_FILTER_SIZE = 0
MORPH_OPEN_ITERATIONS = 0
MORPH_CLOSE_ITERATIONS = 0

RDP_EPSILON_PX = 6.0
MAX_CONTOUR_POINTS = 6000
# Reuse the official OpenMV Canny example (ported in
# ``openmv_edge_detection.py``) as a secondary check. Threshold segmentation
# still supplies the closed contour; Canny only rejects fitted sides that have
# insufficient real image-edge support.
ENABLE_CANNY_VALIDATION = True
EDGE_SUPPORT_RADIUS_PX = 4
EDGE_SUPPORT_SAMPLE_STEP_PX = 4.0
MINIMUM_EDGE_SUPPORT = 0.30

STABLE_FRAMES = 2
STABLE_VERTEX_TOLERANCE_PX = 6.0
REEMIT_MOVEMENT_PX = 10.0
REEMIT_INTERVAL_MS = 1000
TEMPORAL_SMOOTHING_ALPHA = 0.45
DETECT_EVERY_N_FRAMES = 2
GC_EVERY_FRAMES = 20


# High-contrast overlay for both light pieces and dark background regions.
OUTLINE_SHADOW_COLOR = 0
POLYGON_COLOR = 255
VERTEX_COLOR = 255
TEXT_COLOR = 255
TEXT_SHADOW_COLOR = 0
TEXT_SCALE = 2
EDGE_LABEL_OFFSET_PX = 24
VERTEX_LABEL_OFFSET_PX = 18


# Moore-neighbour order, clockwise in image coordinates.
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
    """Support both modern attrtuple fields and older callable blob fields."""
    value = getattr(obj, name)
    return value() if callable(value) else value


def _threshold_value(threshold):
    # OpenMV firmware builds do not all expose the Otsu threshold in the same
    # form.  Stable releases normally return a threshold object, while some
    # v5 development builds return the gray value directly as an int.
    if isinstance(threshold, (int, float)):
        return int(threshold)

    try:
        return _value_or_call(threshold, "value")
    except (AttributeError, TypeError):
        return threshold[0]


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _pixel_value(image_object, x, y):
    """Read one pixel across OpenMV 4.x and 5.x API variants."""
    point = (int(x), int(y))
    try:
        value = image_object.get_pixel(point)
    except TypeError:
        value = image_object.get_pixel(point[0], point[1])

    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _foreground(mask, x, y):
    pixel = _pixel_value(mask, x, y)
    return pixel is not None and pixel != 0


def _find_nearby_boundary_pixel(mask, point, radius=2):
    center_x = int(point[0])
    center_y = int(point[1])
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            x = center_x + offset_x
            y = center_y + offset_y
            if not _foreground(mask, x, y):
                continue
            for neighbour_x, neighbour_y in _NEIGHBOURS:
                if not _foreground(mask, x + neighbour_x, y + neighbour_y):
                    return x, y
    return None


def _find_nearby_foreground_pixel(mask, point, rect):
    """Return the closest foreground pixel to ``point`` inside ``rect``."""
    center_x = int(point[0])
    center_y = int(point[1])
    rect_x, rect_y, rect_width, rect_height = rect
    x_min = max(0, rect_x)
    y_min = max(0, rect_y)
    x_max = min(mask.width() - 1, rect_x + rect_width - 1)
    y_max = min(mask.height() - 1, rect_y + rect_height - 1)
    maximum_radius = max(rect_width, rect_height)

    for radius in range(maximum_radius + 1):
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


def _find_blob_boundary_seed(mask, blob, rect):
    """Find this blob's left boundary, even when blob rectangles overlap.

    Scanning an entire bounding box from its top-left can select a pixel from
    another component whose bounding box overlaps it. Starting at the blob's
    own centroid first identifies its component; walking left then produces
    the boundary/backtrack pair expected by the Moore tracer.
    """
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
        anchor = _find_nearby_foreground_pixel(mask, center, rect)
    if anchor is None:
        return None

    x, y = anchor
    rect_left = max(0, rect[0])
    while x > rect_left and _foreground(mask, x - 1, y):
        x -= 1
    return x, y


def trace_outer_boundary(mask, start, maximum_steps):
    """Trace one connected component's outer boundary using Moore neighbours."""
    # ``_find_blob_boundary_seed`` deliberately returns a point whose west
    # neighbour is background. Preserve that exact point so the initial Moore
    # backtrack is valid. Retain the nearby search for direct/legacy callers.
    if start is None:
        return None, False
    if (
        not _foreground(mask, start[0], start[1])
        or _foreground(mask, start[0] - 1, start[1])
    ):
        start = _find_nearby_boundary_pixel(mask, start)
    if start is None:
        return None, False

    current = start
    backtrack = (start[0] - 1, start[1])
    first_next = None
    contour = [start]

    for _ in range(maximum_steps):
        relative_backtrack = (
            backtrack[0] - current[0],
            backtrack[1] - current[1],
        )
        try:
            backtrack_index = _NEIGHBOURS.index(relative_backtrack)
        except ValueError:
            backtrack_index = 7

        next_point = None
        next_index = None
        for step in range(1, 9):
            neighbour_index = (backtrack_index + step) % 8
            offset_x, offset_y = _NEIGHBOURS[neighbour_index]
            candidate = (current[0] + offset_x, current[1] + offset_y)
            if _foreground(mask, candidate[0], candidate[1]):
                next_point = candidate
                next_index = neighbour_index
                break

        if next_point is None:
            return contour, False

        if first_next is None:
            first_next = next_point
        elif current == start and next_point == first_next:
            if len(contour) > 1 and contour[-1] == start:
                contour.pop()
            return contour, len(contour) >= 3

        previous_index = (next_index - 1) % 8
        previous_offset = _NEIGHBOURS[previous_index]
        backtrack = (
            current[0] + previous_offset[0],
            current[1] + previous_offset[1],
        )
        current = next_point
        contour.append(current)

    return contour, False


def _border_mean(grayscale):
    """Estimate the dominant sheet/background brightness from image borders."""
    width = grayscale.width()
    height = grayscale.height()
    inset_x = min(4, max(0, width // 16))
    inset_y = min(4, max(0, height // 16))
    x_positions = (
        inset_x,
        width // 4,
        width // 2,
        (width * 3) // 4,
        max(0, width - 1 - inset_x),
    )
    y_positions = (
        inset_y,
        height // 4,
        height // 2,
        (height * 3) // 4,
        max(0, height - 1 - inset_y),
    )

    total = 0
    count = 0
    for x in x_positions:
        for y in (inset_y, max(0, height - 1 - inset_y)):
            value = _pixel_value(grayscale, x, y)
            if value is not None:
                total += value
                count += 1
    for y in y_positions[1:-1]:
        for x in (inset_x, max(0, width - 1 - inset_x)):
            value = _pixel_value(grayscale, x, y)
            if value is not None:
                total += value
                count += 1
    return (float(total) / count) if count else 127.5


def _prepare_detection_maps(frame):
    """Build the fixed-camera foreground mask and optional Canny map."""
    try:
        grayscale = frame.copy()
    except Exception as error:
        raise RuntimeError("copy: %s" % str(error))
    try:
        if MEDIAN_FILTER_SIZE > 0:
            grayscale.median(MEDIAN_FILTER_SIZE)
    except Exception as error:
        raise RuntimeError("median: %s" % str(error))
    automatic_threshold = None
    background_mean = None
    use_fixed_threshold = (
        FIXED_FOREGROUND_THRESHOLD is not None
        and FOREGROUND_POLARITY in ("bright", "dark")
    )
    if not use_fixed_threshold:
        try:
            automatic_threshold = _threshold_value(
                grayscale.get_histogram().get_threshold()
            )
        except Exception as error:
            raise RuntimeError("threshold: %s" % str(error))
        try:
            background_mean = _border_mean(grayscale)
        except Exception as error:
            raise RuntimeError("border_mean: %s" % str(error))
    edges = None
    if ENABLE_CANNY_VALIDATION:
        try:
            edges = grayscale.copy()
            # Reuse the verified OpenMV port of 04.Detecting/04.edges.py.
            detect_edges_in_place(edges)
        except Exception as error:
            raise RuntimeError("canny: %s" % str(error))
    mask = grayscale

    if FOREGROUND_POLARITY == "dark":
        foreground_is_dark = True
    elif FOREGROUND_POLARITY == "bright":
        foreground_is_dark = False
    elif abs(background_mean - automatic_threshold) <= 3:
        foreground_is_dark = background_mean > 127
    else:
        foreground_is_dark = background_mean > automatic_threshold

    try:
        if foreground_is_dark:
            polarity = "dark"
            if use_fixed_threshold:
                cutoff = int(FIXED_FOREGROUND_THRESHOLD)
            else:
                cutoff = _clamp(
                    int(automatic_threshold) - OTSU_MARGIN,
                    MIN_AUTO_THRESHOLD,
                    MAX_AUTO_THRESHOLD,
                )
            mask.binary([(0, cutoff)])
        else:
            polarity = "bright"
            if use_fixed_threshold:
                cutoff = int(FIXED_FOREGROUND_THRESHOLD)
            else:
                cutoff = _clamp(
                    int(automatic_threshold) + OTSU_MARGIN,
                    MIN_AUTO_THRESHOLD,
                    MAX_AUTO_THRESHOLD,
                )
            mask.binary([(cutoff, 255)])
    except Exception as error:
        raise RuntimeError("binary: %s" % str(error))

    try:
        for _ in range(MORPH_OPEN_ITERATIONS):
            mask.erode(1)
            mask.dilate(1)
        for _ in range(MORPH_CLOSE_ITERATIONS):
            mask.dilate(1)
            mask.erode(1)
    except Exception as error:
        raise RuntimeError("morphology: %s" % str(error))
    return mask, edges, cutoff, polarity


def _blob_rect(blob):
    return tuple(int(value) for value in _value_or_call(blob, "rect"))


def _blob_perimeter(blob):
    return int(_value_or_call(blob, "perimeter"))


def _blob_pixels(blob):
    return int(_value_or_call(blob, "pixels"))


def _touches_frame_border(rect, width, height):
    x, y, w, h = rect
    margin = FRAME_BORDER_MARGIN_PX
    return (
        x <= margin
        or y <= margin
        or (x + w) >= (width - margin)
        or (y + h) >= (height - margin)
    )


def _candidate_error(reason, rect, details=None):
    result = {"reason": reason, "rect": list(rect)}
    if details is not None:
        result["details"] = details
    return result


def _edge_pixel_near(edges, x, y, radius):
    center_x = int(round(x))
    center_y = int(round(y))
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            value = _pixel_value(
                edges, center_x + offset_x, center_y + offset_y
            )
            if value is not None and value != 0:
                return True
    return False


def polygon_edge_support(
    edges,
    vertices,
    radius=EDGE_SUPPORT_RADIUS_PX,
    sample_step=EDGE_SUPPORT_SAMPLE_STEP_PX,
):
    """Return the fraction of fitted boundary samples supported by Canny."""
    hits = 0
    samples = 0
    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        length = geometry.distance(start, end)
        steps = max(1, int(length / sample_step))
        for step in range(steps + 1):
            ratio = float(step) / steps
            x = start[0] + ((end[0] - start[0]) * ratio)
            y = start[1] + ((end[1] - start[1]) * ratio)
            samples += 1
            if _edge_pixel_near(edges, x, y, radius):
                hits += 1
    return (float(hits) / samples) if samples else 0.0


def detect_polygons(frame):
    try:
        mask, edges, threshold, polarity = _prepare_detection_maps(frame)
    except Exception as error:
        raise RuntimeError("prepare_maps: %s" % str(error))

    width = frame.width()
    height = frame.height()
    try:
        blobs = mask.find_blobs(
            [(1, 255)],
            x_stride=1,
            y_stride=1,
            area_threshold=MIN_BLOB_BOUNDING_AREA,
            pixels_threshold=MIN_BLOB_PIXELS,
            merge=False,
        )
    except Exception as error:
        raise RuntimeError("find_blobs: %s" % str(error))
    blobs = sorted(blobs, key=_blob_pixels, reverse=True)

    rejected = []
    if len(blobs) > MAX_CANDIDATE_BLOBS:
        rejected.append(
            _candidate_error(
                "too_many_candidates",
                (0, 0, width, height),
                len(blobs),
            )
        )
    blobs_to_process = blobs[:MAX_CANDIDATE_BLOBS]
    polygons = []

    for blob in blobs_to_process:
        rect = _blob_rect(blob)
        if _touches_frame_border(rect, width, height):
            # This is normally the sheet/table background, not a puzzle piece.
            continue

        seed = _find_blob_boundary_seed(mask, blob, rect)
        if seed is None:
            rejected.append(
                _candidate_error("missing_boundary_seed", rect)
            )
            continue

        maximum_steps = min(
            MAX_CONTOUR_POINTS,
            max(100, _blob_perimeter(blob) * 4),
        )
        contour, closed = trace_outer_boundary(
            mask, seed, maximum_steps
        )
        if not closed:
            rejected.append(
                _candidate_error(
                    "boundary_trace_failed",
                    rect,
                    0 if contour is None else len(contour),
                )
            )
            continue

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=RDP_EPSILON_PX,
            minimum_edge_length=MINIMUM_VALIDATED_EDGE_PX,
            minimum_area=MIN_POLYGON_AREA_PX,
        )
        if reason is not None:
            rejected.append(
                _candidate_error(
                    reason,
                    rect,
                    {"detected_vertices": len(vertices)},
                )
            )
            continue

        edge_support = None
        if edges is not None:
            edge_support = polygon_edge_support(edges, vertices)
            if edge_support < MINIMUM_EDGE_SUPPORT:
                rejected.append(
                    _candidate_error(
                        "weak_canny_support",
                        rect,
                        round(edge_support, 3),
                    )
                )
                continue

        measurement = geometry.measure_polygon(
            vertices,
            traced_boundary_length_px=geometry.contour_length(contour),
            edge_support=edge_support,
        )
        polygons.append(measurement)

    detected_count = len(polygons)
    polygons.sort(
        key=lambda polygon: (
            polygon["centroid_px"][1],
            polygon["centroid_px"][0],
        )
    )
    for index, polygon in enumerate(polygons):
        polygon["id"] = index + 1

    if detected_count > MAX_POLYGONS:
        status = "too_many_polygons"
        polygons = polygons[:MAX_POLYGONS]
    elif polygons:
        status = "ok"
    elif rejected:
        status = "invalid_shape"
    else:
        status = "no_polygons"

    result = {
        "status": status,
        "threshold": threshold,
        "polarity": polarity,
        "count": len(polygons),
        "polygons": polygons,
        "rejected": rejected,
    }
    return result, mask, edges


def _draw_shadowed_text(frame, x, y, text, color):
    width = frame.width()
    height = frame.height()
    character_width = 8 * TEXT_SCALE
    text_height = 10 * TEXT_SCALE
    x = int(
        _clamp(
            x,
            0,
            max(
                0,
                width
                - (len(text) * character_width)
                - (2 * TEXT_SCALE),
            ),
        )
    )
    y = int(_clamp(y, 0, max(0, height - text_height)))
    frame.draw_string(
        (x + TEXT_SCALE, y + TEXT_SCALE),
        text,
        color=TEXT_SHADOW_COLOR,
        scale=TEXT_SCALE,
    )
    frame.draw_string(
        (x, y), text, color=color, scale=TEXT_SCALE
    )


def _draw_polygon(frame, polygon):
    vertices = polygon["vertices_px"]
    edge_lengths = polygon["edge_lengths_px"]
    center_x, center_y = polygon["centroid_px"]

    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        line = (start[0], start[1], end[0], end[1])
        frame.draw_line(
            line,
            color=OUTLINE_SHADOW_COLOR,
            thickness=8,
        )
        frame.draw_line(
            line,
            color=POLYGON_COLOR,
            thickness=4,
        )

        midpoint_x = (start[0] + end[0]) // 2
        midpoint_y = (start[1] + end[1]) // 2
        direction_x = end[0] - start[0]
        direction_y = end[1] - start[1]
        direction_length = max(
            1.0,
            math.sqrt(
                (direction_x * direction_x)
                + (direction_y * direction_y)
            ),
        )
        normal_x = -direction_y / direction_length
        normal_y = direction_x / direction_length
        if (
            (normal_x * (midpoint_x - center_x))
            + (normal_y * (midpoint_y - center_y))
        ) < 0:
            normal_x = -normal_x
            normal_y = -normal_y

        edge_label = "E%d %.1fPX" % (
            index + 1,
            edge_lengths[index],
        )
        _draw_shadowed_text(
            frame,
            midpoint_x
            + (normal_x * EDGE_LABEL_OFFSET_PX)
            - ((len(edge_label) * 8 * TEXT_SCALE) // 2),
            midpoint_y
            + (normal_y * EDGE_LABEL_OFFSET_PX)
            - (5 * TEXT_SCALE),
            edge_label,
            TEXT_COLOR,
        )

    for index, vertex in enumerate(vertices):
        frame.draw_circle(
            (vertex[0], vertex[1], 8),
            color=OUTLINE_SHADOW_COLOR,
            thickness=6,
        )
        frame.draw_circle(
            (vertex[0], vertex[1], 6),
            color=VERTEX_COLOR,
            thickness=4,
        )
        label = "V%d(%d,%d)" % (
            index + 1,
            vertex[0],
            vertex[1],
        )
        outward_x = vertex[0] - center_x
        outward_y = vertex[1] - center_y
        outward_length = max(
            1.0,
            math.sqrt(
                (outward_x * outward_x)
                + (outward_y * outward_y)
            ),
        )
        label_anchor_x = vertex[0] + (
            (outward_x * VERTEX_LABEL_OFFSET_PX) / outward_length
        )
        label_anchor_y = vertex[1] + (
            (outward_y * VERTEX_LABEL_OFFSET_PX) / outward_length
        )
        if outward_x < 0:
            label_x = (
                label_anchor_x
                - (len(label) * 8 * TEXT_SCALE)
                - (2 * TEXT_SCALE)
            )
        else:
            label_x = label_anchor_x + (2 * TEXT_SCALE)
        _draw_shadowed_text(
            frame,
            label_x,
            label_anchor_y - (5 * TEXT_SCALE),
            label,
            TEXT_COLOR,
        )

def draw_result(frame, result):
    for index, polygon in enumerate(result["polygons"]):
        _draw_shadowed_text(
            frame,
            4,
            4 + (index * 20),
            "P%d: %d SIDES" % (
                polygon["id"],
                polygon["vertex_count"],
            ),
            TEXT_COLOR,
        )
        _draw_polygon(frame, polygon)


def _results_close(first, second, tolerance):
    if first is None or second is None:
        return False
    if first["status"] != second["status"]:
        return False
    if first["count"] != second["count"]:
        return False

    if first["status"] != "ok":
        first_errors = sorted(
            error["reason"] for error in first["rejected"]
        )
        second_errors = sorted(
            error["reason"] for error in second["rejected"]
        )
        if first_errors != second_errors:
            return False

    for first_polygon, second_polygon in zip(
        first["polygons"], second["polygons"]
    ):
        first_vertices = first_polygon["vertices_px"]
        second_vertices = second_polygon["vertices_px"]
        if len(first_vertices) != len(second_vertices):
            return False
        for first_vertex, second_vertex in zip(
            first_vertices, second_vertices
        ):
            dx = first_vertex[0] - second_vertex[0]
            dy = first_vertex[1] - second_vertex[1]
            if math.sqrt((dx * dx) + (dy * dy)) > tolerance:
                return False
    return True


def _serializable_polygon(polygon):
    return {
        "id": polygon["id"],
        "side_count": polygon["vertex_count"],
        "vertices_px": polygon["vertices_px"],
    }


def make_serial_payload(result):
    return {
        "status": result["status"],
        "count": result["count"],
        "coordinate_system": {
            "origin_px": [0, 0],
            "x_direction": "right",
            "y_direction": "down",
            "vertex_order": "clockwise",
            "first_vertex": "topmost_then_leftmost",
        },
        "polygons": [
            _serializable_polygon(polygon)
            for polygon in result["polygons"]
        ],
    }


def is_simulator_payload(payload):
    """Return True only for records accepted by the desktop simulator."""
    polygons = payload.get("polygons")
    return (
        payload.get("status") == "ok"
        and isinstance(polygons, list)
        and 1 <= len(polygons) <= MAX_POLYGONS
        and payload.get("count") == len(polygons)
    )


def _smooth_result(previous, current):
    """Low-pass stable vertex coordinates and recompute pixel measurements."""
    if (
        previous is None
        or current["status"] != "ok"
        or not _results_close(
            previous,
            current,
            STABLE_VERTEX_TOLERANCE_PX,
        )
    ):
        return current

    smoothed = dict(current)
    smoothed_polygons = []
    alpha = TEMPORAL_SMOOTHING_ALPHA
    for old_polygon, new_polygon in zip(
        previous["polygons"], current["polygons"]
    ):
        vertices = []
        for old_vertex, new_vertex in zip(
            old_polygon["vertices_px"], new_polygon["vertices_px"]
        ):
            vertices.append(
                (
                    old_vertex[0]
                    + ((new_vertex[0] - old_vertex[0]) * alpha),
                    old_vertex[1]
                    + ((new_vertex[1] - old_vertex[1]) * alpha),
                )
            )

        measurement = geometry.measure_polygon(
            vertices,
            traced_boundary_length_px=new_polygon[
                "traced_boundary_length_px"
            ],
            edge_support=new_polygon["edge_support"],
        )
        measurement["id"] = new_polygon["id"]
        smoothed_polygons.append(measurement)

    smoothed["polygons"] = smoothed_polygons
    return smoothed


def _ticks_ms():
    ticks_ms = getattr(time, "ticks_ms", None)
    if ticks_ms is not None:
        return ticks_ms()
    return int(time.time() * 1000)


def _ticks_diff(new_ticks, old_ticks):
    ticks_diff = getattr(time, "ticks_diff", None)
    if ticks_diff is not None:
        return ticks_diff(new_ticks, old_ticks)
    return new_ticks - old_ticks


class StableResultReporter:
    def __init__(self, uart):
        self.uart = uart
        self.pending_result = None
        self.pending_frames = 0
        self.stable_result = None
        self.last_emitted_result = None
        self.last_emitted_ms = None

    def update(self, result):
        if _results_close(
            result,
            self.pending_result,
            STABLE_VERTEX_TOLERANCE_PX,
        ):
            self.pending_frames += 1
            self.pending_result = _smooth_result(
                self.pending_result, result
            )
        else:
            self.pending_result = result
            self.pending_frames = 1

        if self.pending_frames < STABLE_FRAMES:
            return self.stable_result

        self.stable_result = self.pending_result

        now_ms = _ticks_ms()
        result_unchanged = _results_close(
            self.stable_result,
            self.last_emitted_result,
            REEMIT_MOVEMENT_PX,
        )
        if (
            result_unchanged
            and self.last_emitted_ms is not None
            and _ticks_diff(now_ms, self.last_emitted_ms)
            < REEMIT_INTERVAL_MS
        ):
            return self.stable_result

        payload = make_serial_payload(self.stable_result)
        if not is_simulator_payload(payload):
            return self.stable_result

        message = json.dumps(payload)
        self.uart.write(message)
        self.uart.write("\r\n")
        self.last_emitted_result = self.stable_result
        self.last_emitted_ms = now_ms
        return self.stable_result


def _processing_error_result(error):
    return {
        "status": "processing_error",
        "threshold": -1,
        "polarity": "unknown",
        "count": 0,
        "polygons": [],
        "rejected": [
            {
                "reason": "processing_error",
                "rect": [0, 0, 0, 0],
                "details": str(error),
            }
        ],
    }


def main():
    # OpenMV H7 Plus UART3: TX=P4, RX=P5. Only TX is required by the bridge.
    from machine import UART

    uart = UART(3, baudrate=115200, bits=8, parity=None, stop=1)
    # Clear any partial/noise record accumulated while P4 was undriven.
    uart.write("\r\n")

    # Reuse the camera and buffering setup already proven by 04.edges.py.
    camera = init_camera()
    locked_gain_db = None
    locked_exposure_us = None
    if LOCK_CAMERA_SETTINGS:
        try:
            locked_gain_db = camera.gain_db()
            locked_exposure_us = camera.exposure_us()
            camera.auto_gain(False, gain_db=locked_gain_db)
            camera.auto_exposure(
                False, exposure_us=locked_exposure_us
            )
            camera.snapshot(time=250)
        except Exception as error:
            print("CAMERA_LOCK_WARNING: %s" % str(error))

    clock = time.clock()
    reporter = StableResultReporter(uart)
    frame_number = 0

    print("POLYGON_DETECTOR_READY")
    print(
        "EDGE_VALIDATION=%s MAX_POLYGONS=%d SIDES=3..5"
        % (
            "canny" if ENABLE_CANNY_VALIDATION else "disabled",
            MAX_POLYGONS,
        )
    )
    print("DISPLAY=GRAYSCALE_WITH_POLYGON_OVERLAY UNITS=PX")
    print(
        "MIN_EDGE_TARGET=%.1fPX VALIDATE_AT=%.1fPX"
        % (MINIMUM_EDGE_PX, MINIMUM_VALIDATED_EDGE_PX)
    )
    if locked_gain_db is not None and locked_exposure_us is not None:
        print(
            "CAMERA_LOCKED gain=%.2fdB exposure=%dus"
            % (locked_gain_db, locked_exposure_us)
        )

    try:
        while True:
            clock.tick()
            frame = camera.snapshot()
            should_detect = (
                reporter.stable_result is None
                or frame_number % DETECT_EVERY_N_FRAMES == 0
            )
            if should_detect:
                try:
                    result, mask, edges = detect_polygons(frame)
                    del mask
                    del edges
                except Exception as error:
                    result = _processing_error_result(error)
                display_result = reporter.update(result)
            else:
                display_result = reporter.stable_result

            if display_result is not None:
                draw_result(frame, display_result)
            # Double buffering can otherwise let the IDE request the next raw
            # buffer before the overlay is transferred. Flush the completed
            # annotated grayscale frame explicitly.
            camera.flush()

            frame_number += 1
            if frame_number % GC_EVERY_FRAMES == 0:
                gc.collect()
    except KeyboardInterrupt:
        camera.flush()
        print("POLYGON_DETECTOR_STOPPED")


if __name__ == "__main__":
    main()
