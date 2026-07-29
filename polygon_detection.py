"""Detect 1-4 bright three-to-five-sided polygons on a dark background.

Target runtime: CanMV K230 v1.4+ using the media camera API. The annotated RGB
frame is shown in the CanMV IDE preview. Stable results are emitted as one JSON
object per line over the IDE/USB serial console.
"""

import gc
import math
import os
import time

from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

try:
    import json
except ImportError:
    import ujson as json

import polygon_geometry as geometry


# ---------------------------------------------------------------------------
# Camera and calibration configuration
# ---------------------------------------------------------------------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_FPS = 30
CAMERA_WARMUP_MS = 2000

# Set this after fixing the camera height. For example, if a 10 cm reference
# measures 153 pixels in the image, set PIXELS_PER_CM = 15.3.
# None deliberately disables centimetre output instead of reporting bad units.
PIXELS_PER_CM = None

# The centimetre check is intentionally tolerant because the official pieces
# are guaranteed to have edges >= 2 cm and pixel fitting has small errors.
MINIMUM_ACCEPTED_EDGE_CM = 1.5
MINIMUM_EDGE_PX_WITHOUT_CALIBRATION = 12.0


# ---------------------------------------------------------------------------
# Segmentation and polygon configuration
# ---------------------------------------------------------------------------

MAX_POLYGONS = 4
MAX_CANDIDATE_BLOBS = 8
MIN_BLOB_PIXELS = 80
MIN_BLOB_BOUNDING_AREA = 100
MIN_POLYGON_AREA_PX = 80.0
FRAME_BORDER_MARGIN_PX = 2

OTSU_MARGIN = 5
MIN_AUTO_THRESHOLD = 20
MAX_AUTO_THRESHOLD = 235
MEDIAN_FILTER_SIZE = 1
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 1

RDP_EPSILON_PX = 3.0
MAX_CONTOUR_POINTS = 3000

STABLE_FRAMES = 3
STABLE_VERTEX_TOLERANCE_PX = 3.0
REEMIT_MOVEMENT_PX = 6.0
GC_EVERY_FRAMES = 20


# RGB888 colors accepted by OpenMV RGB565 drawing methods.
POLYGON_COLORS = (
    (0, 255, 0),
    (0, 180, 255),
    (255, 220, 0),
    (255, 0, 255),
    (0, 255, 180),
    (255, 128, 0),
    (180, 180, 255),
    (255, 80, 80),
)
ERROR_COLOR = (255, 0, 0)
TEXT_SHADOW_COLOR = (0, 0, 0)
HEADER_TEXT_COLOR = (255, 255, 255)
CALIBRATION_WARNING_COLOR = (255, 210, 0)


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
    try:
        return _value_or_call(threshold, "value")
    except (AttributeError, TypeError):
        return threshold[0]


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _foreground(mask, x, y):
    pixel = mask.get_pixel(x, y)
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


def trace_outer_boundary(mask, start, maximum_steps):
    """Trace one connected component's outer boundary using Moore neighbours."""
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


def _prepare_binary_mask(frame):
    mask = frame.to_grayscale(copy=True)
    if MEDIAN_FILTER_SIZE > 0:
        mask.median(MEDIAN_FILTER_SIZE)

    automatic_threshold = _threshold_value(
        mask.get_histogram().get_threshold()
    )
    cutoff = _clamp(
        int(automatic_threshold) + OTSU_MARGIN,
        MIN_AUTO_THRESHOLD,
        MAX_AUTO_THRESHOLD,
    )
    mask.binary([(cutoff, 255)])

    for _ in range(MORPH_OPEN_ITERATIONS):
        mask.erode(1)
        mask.dilate(1)
    for _ in range(MORPH_CLOSE_ITERATIONS):
        mask.dilate(1)
        mask.erode(1)
    return mask, cutoff


def _blob_rect(blob):
    return tuple(int(value) for value in _value_or_call(blob, "rect"))


def _blob_corners(blob):
    return _value_or_call(blob, "corners")


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


def _minimum_edge_pixels():
    if PIXELS_PER_CM is None or PIXELS_PER_CM <= 0:
        return MINIMUM_EDGE_PX_WITHOUT_CALIBRATION
    return max(
        4.0, float(PIXELS_PER_CM) * MINIMUM_ACCEPTED_EDGE_CM
    )


def _candidate_error(reason, rect, details=None):
    result = {"reason": reason, "rect": list(rect)}
    if details is not None:
        result["details"] = details
    return result


def detect_polygons(frame):
    mask, threshold = _prepare_binary_mask(frame)
    width = frame.width()
    height = frame.height()
    blobs = mask.find_blobs(
        [(200, 255)],
        x_stride=1,
        y_stride=1,
        area_threshold=MIN_BLOB_BOUNDING_AREA,
        pixels_threshold=MIN_BLOB_PIXELS,
        merge=False,
    )
    blobs = sorted(blobs, key=_blob_pixels, reverse=True)

    errors = []
    if len(blobs) > MAX_CANDIDATE_BLOBS:
        errors.append(
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
            errors.append(_candidate_error("touches_frame_border", rect))
            continue

        corners = _blob_corners(blob)
        if not corners:
            errors.append(_candidate_error("missing_boundary_seed", rect))
            continue

        maximum_steps = min(
            MAX_CONTOUR_POINTS,
            max(100, _blob_perimeter(blob) * 4),
        )
        contour, closed = trace_outer_boundary(
            mask, corners[0], maximum_steps
        )
        if not closed:
            errors.append(
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
            minimum_edge_length=_minimum_edge_pixels(),
            minimum_area=MIN_POLYGON_AREA_PX,
        )
        if reason is not None:
            errors.append(
                _candidate_error(
                    reason,
                    rect,
                    {"detected_vertices": len(vertices)},
                )
            )
            continue

        measurement = geometry.measure_polygon(
            vertices, pixels_per_cm=PIXELS_PER_CM
        )
        polygons.append(measurement)

    polygons.sort(
        key=lambda polygon: (
            polygon["centroid_px"][1],
            polygon["centroid_px"][0],
        )
    )
    for index, polygon in enumerate(polygons):
        polygon["id"] = index + 1

    if len(polygons) > MAX_POLYGONS:
        status = "too_many_polygons"
    elif errors:
        status = "invalid_shape"
    elif not polygons:
        status = "no_polygons"
    else:
        status = "ok"

    result = {
        "status": status,
        "threshold": threshold,
        "count": len(polygons),
        "polygons": polygons,
        "errors": errors,
    }
    return result, mask


def _draw_shadowed_text(frame, x, y, text, color):
    width = frame.width()
    height = frame.height()
    x = int(_clamp(x, 0, max(0, width - (len(text) * 8) - 2)))
    y = int(_clamp(y, 0, max(0, height - 10)))
    frame.draw_string(
        (x + 1, y + 1), text, color=TEXT_SHADOW_COLOR, scale=1
    )
    frame.draw_string((x, y), text, color=color, scale=1)


def _draw_polygon(frame, polygon, color):
    vertices = polygon["vertices_px"]
    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        frame.draw_line(
            (start[0], start[1], end[0], end[1]),
            color=color,
            thickness=2,
        )

    for index, vertex in enumerate(vertices):
        frame.draw_circle(
            (vertex[0], vertex[1], 3), color=color, thickness=2
        )
        label = "%d.%d(%d,%d)" % (
            polygon["id"],
            index + 1,
            vertex[0],
            vertex[1],
        )
        label_y = vertex[1] - 11 if vertex[1] >= 14 else vertex[1] + 5
        _draw_shadowed_text(
            frame, vertex[0] + 4, label_y, label, color
        )

    center_x, center_y = polygon["centroid_px"]
    if polygon["perimeter_cm"] is None:
        perimeter_label = "#%d P=%.1fpx" % (
            polygon["id"],
            polygon["perimeter_px"],
        )
    else:
        perimeter_label = "#%d P=%.2fcm" % (
            polygon["id"],
            polygon["perimeter_cm"],
        )
    _draw_shadowed_text(
        frame, center_x - 25, center_y - 5, perimeter_label, color
    )


def draw_result(frame, result, fps):
    for index, polygon in enumerate(result["polygons"]):
        color = POLYGON_COLORS[index % len(POLYGON_COLORS)]
        _draw_polygon(frame, polygon, color)

    for error in result["errors"]:
        rect = tuple(error["rect"])
        if rect[2] > 0 and rect[3] > 0:
            frame.draw_rectangle(rect, color=ERROR_COLOR, thickness=2)
        _draw_shadowed_text(
            frame,
            rect[0],
            rect[1] + rect[3] + 2,
            error["reason"],
            ERROR_COLOR,
        )

    frame.draw_rectangle(
        (0, 0, frame.width(), 23),
        color=TEXT_SHADOW_COLOR,
        fill=True,
    )
    header = "N:%d T:%d FPS:%.1f %s" % (
        result["count"],
        result["threshold"],
        fps,
        result["status"],
    )
    frame.draw_string(
        (1, 1), header, color=HEADER_TEXT_COLOR, scale=1
    )
    if PIXELS_PER_CM is None or PIXELS_PER_CM <= 0:
        frame.draw_string(
            (1, 12),
            "CALIBRATION REQUIRED: set PIXELS_PER_CM",
            color=CALIBRATION_WARNING_COLOR,
            scale=1,
        )
    else:
        frame.draw_string(
            (1, 12),
            "scale=%.3f px/cm" % PIXELS_PER_CM,
            color=HEADER_TEXT_COLOR,
            scale=1,
        )


def _results_close(first, second, tolerance):
    if first is None or second is None:
        return False
    if first["status"] != second["status"]:
        return False
    if first["count"] != second["count"]:
        return False

    first_errors = sorted(error["reason"] for error in first["errors"])
    second_errors = sorted(error["reason"] for error in second["errors"])
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
        "vertices_px": polygon["vertices_px"],
        "vertices_cm": polygon["vertices_cm"],
        "edge_lengths_px": polygon["edge_lengths_px"],
        "edge_lengths_cm": polygon["edge_lengths_cm"],
        "perimeter_px": polygon["perimeter_px"],
        "perimeter_cm": polygon["perimeter_cm"],
        "centroid_px": polygon["centroid_px"],
    }


def make_serial_payload(result):
    return {
        "status": result["status"],
        "count": result["count"],
        "threshold": result["threshold"],
        "scale_ready": PIXELS_PER_CM is not None and PIXELS_PER_CM > 0,
        "pixels_per_cm": PIXELS_PER_CM,
        "polygons": [
            _serializable_polygon(polygon)
            for polygon in result["polygons"]
        ],
        "errors": result["errors"],
    }


class StableResultReporter:
    def __init__(self):
        self.pending_result = None
        self.pending_frames = 0
        self.last_emitted_result = None

    def update(self, result):
        if _results_close(
            result,
            self.pending_result,
            STABLE_VERTEX_TOLERANCE_PX,
        ):
            self.pending_frames += 1
        else:
            self.pending_result = result
            self.pending_frames = 1

        if self.pending_frames < STABLE_FRAMES:
            return False

        if _results_close(
            result,
            self.last_emitted_result,
            REEMIT_MOVEMENT_PX,
        ):
            return False

        print(json.dumps(make_serial_payload(result)))
        self.last_emitted_result = result
        return True


def _processing_error_result(error):
    return {
        "status": "processing_error",
        "threshold": -1,
        "count": 0,
        "polygons": [],
        "errors": [
            {
                "reason": "processing_error",
                "rect": [0, 0, 0, 0],
                "details": str(error),
            }
        ],
    }


def main():
    sensor = None
    display_initialized = False
    media_initialized = False

    try:
        sensor = Sensor(width=FRAME_WIDTH, height=FRAME_HEIGHT)
        sensor.reset()
        sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)
        sensor.set_pixformat(Sensor.RGB565)

        # VIRT sends the annotated frame to CanMV IDE without requiring a
        # particular LCD panel or HDMI monitor.
        Display.init(
            Display.VIRT,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=DISPLAY_FPS,
        )
        display_initialized = True

        MediaManager.init()
        media_initialized = True
        sensor.run()
        time.sleep_ms(CAMERA_WARMUP_MS)

        clock = time.clock()
        reporter = StableResultReporter()
        frame_number = 0

        print("POLYGON_DETECTOR_READY")
        if PIXELS_PER_CM is None or PIXELS_PER_CM <= 0:
            print("CALIBRATION_REQUIRED: set PIXELS_PER_CM")

        while True:
            os.exitpoint()
            clock.tick()
            frame = sensor.snapshot()
            try:
                result, mask = detect_polygons(frame)
                del mask
            except Exception as error:
                result = _processing_error_result(error)

            draw_result(frame, result, clock.fps())
            reporter.update(result)
            Display.show_image(frame)

            frame_number += 1
            if frame_number % GC_EVERY_FRAMES == 0:
                gc.collect()
    except KeyboardInterrupt:
        print("POLYGON_DETECTOR_STOPPED")
    except BaseException as error:
        print("POLYGON_DETECTOR_FATAL:", error)
        raise
    finally:
        if isinstance(sensor, Sensor):
            try:
                sensor.stop()
            except BaseException as error:
                print("SENSOR_STOP_WARNING:", error)
        if display_initialized:
            try:
                Display.deinit()
            except BaseException as error:
                print("DISPLAY_DEINIT_WARNING:", error)

        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)

        if media_initialized:
            try:
                MediaManager.deinit()
            except BaseException as error:
                print("MEDIA_DEINIT_WARNING:", error)


if __name__ == "__main__":
    main()
