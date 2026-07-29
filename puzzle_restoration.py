"""Pure-Python geometry and restoration planning for OpenMV puzzle data.

The module deliberately avoids NumPy and OpenCV so it can be used by the
desktop simulator with only the Python standard library.  Input polygons use
the same image coordinate system as ``polygon_detection.py``: origin at the
top-left, X right, Y down, vertices in boundary order.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import statistics
from typing import Iterable, Sequence


Point = tuple[float, float]
# x' = a*x + b*y + tx, y' = c*x + d*y + ty
Transform = tuple[float, float, float, float, float, float]

_EPSILON = 1e-8


class PuzzleSolveError(RuntimeError):
    """Raised when camera data cannot be parsed or assembled."""


@dataclass(frozen=True)
class Piece:
    piece_id: int
    vertices: tuple[Point, ...]


@dataclass(frozen=True)
class EdgeMatch:
    relative_error: float
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int


@dataclass(frozen=True)
class MotionPlan:
    piece_id: int
    source_center: Point
    target_center: Point
    delta_angle_deg: float
    transform: Transform


@dataclass(frozen=True)
class FixedPieceTemplate:
    template_id: str
    name: str
    vertices_cm: tuple[Point, ...]


@dataclass(frozen=True)
class FixedTemplateMatch:
    piece_id: int
    template_id: str
    template_name: str
    normalized_error: float
    pixels_per_cm: float
    vertex_shift: int
    confidence: str
    observation_rms_cm: float
    observation_max_cm: float


@dataclass(frozen=True)
class _TemplateFit:
    template_index: int
    aligned_vertices_cm: tuple[Point, ...]
    vertex_shift: int
    pixels_per_cm: float
    normalized_error: float


@dataclass(frozen=True)
class PuzzleSolution:
    pieces: tuple[Piece, ...]
    transforms: tuple[Transform, ...]
    matches: tuple[EdgeMatch, ...]
    motions: tuple[MotionPlan, ...]
    target_bounds: tuple[float, float, float, float]
    score: float
    mode: str = "general"
    template_matches: tuple[FixedTemplateMatch, ...] = ()
    pixels_per_cm: float | None = None
    target_polygons: tuple[tuple[Point, ...], ...] = ()
    warnings: tuple[str, ...] = ()
    clearance_cm: float = 0.0


# PDF question 1, Figure 2:
# - outer rectangle: 10 cm x 6 cm
# - top split: 2 cm + 8 cm
# - left split: 2 cm + 1 cm + 3 cm
# - diagonal from (2, 0) to (10, 6) is 10 cm
# - first diagonal segment is 2 cm; last diagonal segment is 3 cm
#
# Consequently H=(3.6, 1.2) and I=(7.6, 4.2). Collinear junctions H/I
# are not outline vertices of the isolated upper-right triangle.
QUESTION_ONE_TEMPLATES: tuple[FixedPieceTemplate, ...] = (
    FixedPieceTemplate(
        "F1",
        "左上四边形",
        ((0.0, 0.0), (2.0, 0.0), (3.6, 1.2), (0.0, 2.0)),
    ),
    FixedPieceTemplate(
        "F2",
        "右上三角形",
        ((2.0, 0.0), (10.0, 0.0), (10.0, 6.0)),
    ),
    FixedPieceTemplate(
        "F3",
        "中部四边形",
        ((0.0, 2.0), (3.6, 1.2), (7.6, 4.2), (0.0, 3.0)),
    ),
    FixedPieceTemplate(
        "F4",
        "下部四边形",
        ((0.0, 3.0), (7.6, 4.2), (10.0, 6.0), (0.0, 6.0)),
    ),
)


def identity_transform() -> Transform:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def rigid_transform(angle_rad: float, tx: float = 0.0, ty: float = 0.0) -> Transform:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (cosine, -sine, sine, cosine, float(tx), float(ty))


def compose_transform(outer: Transform, inner: Transform) -> Transform:
    """Return the transform ``outer @ inner``."""
    oa, ob, oc, od, otx, oty = outer
    ia, ib, ic, id_, itx, ity = inner
    return (
        (oa * ia) + (ob * ic),
        (oa * ib) + (ob * id_),
        (oc * ia) + (od * ic),
        (oc * ib) + (od * id_),
        (oa * itx) + (ob * ity) + otx,
        (oc * itx) + (od * ity) + oty,
    )


def inverse_rigid_transform(transform: Transform) -> Transform:
    """Return the inverse of a rotation+translation transform."""
    a, b, c, d, tx, ty = transform
    return (
        a,
        c,
        b,
        d,
        -((a * tx) + (c * ty)),
        -((b * tx) + (d * ty)),
    )


def apply_transform(point: Point, transform: Transform) -> Point:
    a, b, c, d, tx, ty = transform
    return (
        (a * point[0]) + (b * point[1]) + tx,
        (c * point[0]) + (d * point[1]) + ty,
    )


def transform_polygon(vertices: Sequence[Point], transform: Transform) -> list[Point]:
    return [apply_transform(point, transform) for point in vertices]


def transform_angle(transform: Transform) -> float:
    return math.atan2(transform[2], transform[0])


def signed_area(vertices: Sequence[Point]) -> float:
    total = 0.0
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        total += (first[0] * second[1]) - (second[0] * first[1])
    return total * 0.5


def polygon_area(vertices: Sequence[Point]) -> float:
    return abs(signed_area(vertices))


def polygon_centroid(vertices: Sequence[Point]) -> Point:
    area_twice = 0.0
    x_total = 0.0
    y_total = 0.0
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        cross_value = (first[0] * second[1]) - (second[0] * first[1])
        area_twice += cross_value
        x_total += (first[0] + second[0]) * cross_value
        y_total += (first[1] + second[1]) * cross_value
    if abs(area_twice) < _EPSILON:
        return (
            sum(point[0] for point in vertices) / len(vertices),
            sum(point[1] for point in vertices) / len(vertices),
        )
    scale = 1.0 / (3.0 * area_twice)
    return x_total * scale, y_total * scale


def distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def polygon_edges(vertices: Sequence[Point]) -> list[tuple[Point, Point]]:
    return [
        (vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    ]


def _clearance_translation(
    first_polygon: Sequence[Point],
    second_polygon: Sequence[Point],
    reference_direction: Point,
    clearance_px: float,
) -> Point | None:
    """Return the smallest SAT translation separating two convex polygons.

    The returned vector points from the first reference piece toward the
    second.  Half can be applied to each polygon in opposite directions.  A
    positive clearance treats near-touching polygons as collisions too.
    """
    axes = []
    for polygon in (first_polygon, second_polygon):
        for first, second in polygon_edges(polygon):
            edge_x = second[0] - first[0]
            edge_y = second[1] - first[1]
            length = math.hypot(edge_x, edge_y)
            if length > _EPSILON:
                axes.append((-edge_y / length, edge_x / length))

    best_axis = None
    best_distance = None
    for axis in axes:
        first_projection = [
            (point[0] * axis[0]) + (point[1] * axis[1])
            for point in first_polygon
        ]
        second_projection = [
            (point[0] * axis[0]) + (point[1] * axis[1])
            for point in second_polygon
        ]
        overlap = min(
            max(first_projection), max(second_projection)
        ) - max(min(first_projection), min(second_projection))
        distance_needed = overlap + clearance_px
        if distance_needed <= _EPSILON:
            return None
        if best_distance is None or distance_needed < best_distance:
            best_distance = distance_needed
            best_axis = axis

    if best_axis is None or best_distance is None:
        return None
    if (
        (best_axis[0] * reference_direction[0])
        + (best_axis[1] * reference_direction[1])
    ) < 0.0:
        best_axis = (-best_axis[0], -best_axis[1])
    return best_axis[0] * best_distance, best_axis[1] * best_distance


def _separate_target_polygons(
    polygons: Sequence[Sequence[Point]],
    reference_centers: Sequence[Point],
    clearance_px: float,
    maximum_iterations: int = 160,
) -> tuple[Point, ...]:
    """Find small per-piece translations that remove overlap and add a gap."""
    offsets = [[0.0, 0.0] for _ in polygons]
    for _ in range(maximum_iterations):
        largest_adjustment = 0.0
        for first_index in range(len(polygons)):
            for second_index in range(first_index + 1, len(polygons)):
                first_polygon = [
                    (
                        point[0] + offsets[first_index][0],
                        point[1] + offsets[first_index][1],
                    )
                    for point in polygons[first_index]
                ]
                second_polygon = [
                    (
                        point[0] + offsets[second_index][0],
                        point[1] + offsets[second_index][1],
                    )
                    for point in polygons[second_index]
                ]
                reference_direction = (
                    reference_centers[second_index][0]
                    - reference_centers[first_index][0],
                    reference_centers[second_index][1]
                    - reference_centers[first_index][1],
                )
                translation = _clearance_translation(
                    first_polygon,
                    second_polygon,
                    reference_direction,
                    clearance_px,
                )
                if translation is None:
                    continue
                half_x = translation[0] * 0.5
                half_y = translation[1] * 0.5
                offsets[first_index][0] -= half_x
                offsets[first_index][1] -= half_y
                offsets[second_index][0] += half_x
                offsets[second_index][1] += half_y
                largest_adjustment = max(
                    largest_adjustment,
                    math.hypot(translation[0], translation[1]),
                )
        if largest_adjustment <= 1e-5:
            break
    return tuple((offset[0], offset[1]) for offset in offsets)


def _cross(first: Point, second: Point, third: Point) -> float:
    return (
        ((second[0] - first[0]) * (third[1] - first[1]))
        - ((second[1] - first[1]) * (third[0] - first[0]))
    )


def _normalise_polygon(vertices: Iterable[Sequence[float]]) -> tuple[Point, ...]:
    converted = [(float(point[0]), float(point[1])) for point in vertices]
    if len(converted) > 1 and distance(converted[0], converted[-1]) < _EPSILON:
        converted.pop()
    if not 3 <= len(converted) <= 5:
        raise PuzzleSolveError("每块碎片必须包含 3～5 个顶点")
    if polygon_area(converted) < 1.0:
        raise PuzzleSolveError("检测到面积过小或退化的碎片")
    # Positive signed area is clockwise in the camera's Y-down coordinate view.
    if signed_area(converted) < 0:
        converted.reverse()
    return tuple(converted)


def pieces_from_payload(payload: dict) -> tuple[Piece, ...]:
    """Validate one JSON payload emitted by ``polygon_detection.py``."""
    if payload.get("status") != "ok":
        raise PuzzleSolveError("摄像头状态不是 ok：%s" % payload.get("status"))
    raw_polygons = payload.get("polygons")
    if not isinstance(raw_polygons, list) or not 1 <= len(raw_polygons) <= 4:
        raise PuzzleSolveError("摄像头必须回传 1～4 块碎片")
    pieces = []
    identifiers = set()
    for index, raw_polygon in enumerate(raw_polygons):
        try:
            piece_id = int(raw_polygon.get("id", index + 1))
            vertices = _normalise_polygon(raw_polygon["vertices_px"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise PuzzleSolveError("碎片数据格式错误：%s" % str(error)) from error
        if piece_id in identifiers:
            raise PuzzleSolveError("碎片 ID 重复：%d" % piece_id)
        identifiers.add(piece_id)
        pieces.append(Piece(piece_id, vertices))
    expected_count = payload.get("count")
    if expected_count is not None and int(expected_count) != len(pieces):
        raise PuzzleSolveError("count 与 polygons 数量不一致")
    return tuple(pieces)


def payload_from_json_line(line: str) -> dict:
    """Parse a serial line that may contain OpenMV logging before the JSON."""
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end < start:
        raise PuzzleSolveError("这一行不包含 JSON 数据")
    try:
        value = json.loads(line[start : end + 1])
    except ValueError as error:
        raise PuzzleSolveError("JSON 解析失败：%s" % str(error)) from error
    if not isinstance(value, dict):
        raise PuzzleSolveError("摄像头 JSON 顶层必须是对象")
    return value


def payload_from_text(text: str) -> dict:
    """Return the last complete JSON object from pasted terminal output.

    The input may be one pretty-printed JSON object, or a mixture of OpenMV
    startup logs and multiple newline-delimited payloads.  Returning the latest
    complete object matches what SET should freeze on a live camera.
    """
    if not isinstance(text, str) or not text.strip():
        raise PuzzleSolveError("没有可导入的 JSON 文本")
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return value

    decoder = json.JSONDecoder()
    latest = None
    position = 0
    while position < len(text):
        object_start = text.find("{", position)
        if object_start < 0:
            break
        try:
            candidate, object_end = decoder.raw_decode(text, object_start)
        except ValueError:
            position = object_start + 1
            continue
        if isinstance(candidate, dict):
            latest = candidate
        position = max(object_start + 1, object_end)
    if latest is None:
        raise PuzzleSolveError("粘贴内容中没有找到完整的 JSON 对象")
    return latest


def _align_edge(
    source_a: Point,
    source_b: Point,
    destination_a: Point,
    destination_b: Point,
) -> Transform:
    source_angle = math.atan2(
        source_b[1] - source_a[1], source_b[0] - source_a[0]
    )
    destination_angle = math.atan2(
        destination_b[1] - destination_a[1],
        destination_b[0] - destination_a[0],
    )
    result = rigid_transform(destination_angle - source_angle)
    mapped = apply_transform(source_a, result)
    a, b, c, d, _, _ = result
    return (
        a,
        b,
        c,
        d,
        destination_a[0] - mapped[0],
        destination_a[1] - mapped[1],
    )


def _candidate_matches(
    pieces: Sequence[Piece],
    length_tolerance: float,
    maximum_candidates: int,
) -> list[EdgeMatch]:
    edge_records = []
    for piece_index, piece in enumerate(pieces):
        for edge_index, (first, second) in enumerate(polygon_edges(piece.vertices)):
            edge_records.append(
                (piece_index, edge_index, first, second, distance(first, second))
            )
    candidates = []
    for first, second in itertools.combinations(edge_records, 2):
        if first[0] == second[0]:
            continue
        relative_error = abs(first[4] - second[4]) / max(first[4], second[4])
        if relative_error <= length_tolerance:
            candidates.append(
                EdgeMatch(
                    relative_error,
                    first[0],
                    first[1],
                    second[0],
                    second[1],
                )
            )
    candidates.sort(key=lambda match: match.relative_error)
    return candidates[:maximum_candidates]


def _connected_matching_sets(
    candidates: Sequence[EdgeMatch],
    piece_count: int,
    maximum_layouts: int,
):
    if piece_count == 1:
        yield ()
        return
    emitted = 0
    # Any valid assembly has a spanning tree. Searching only N-1 connections
    # supports chain, tree and cycle puzzles without the old degree-2 limit.
    for combination in itertools.combinations(candidates, piece_count - 1):
        used_edges = set()
        graph = [set() for _ in range(piece_count)]
        valid = True
        for match in combination:
            first_key = (match.piece_a, match.edge_a)
            second_key = (match.piece_b, match.edge_b)
            if first_key in used_edges or second_key in used_edges:
                valid = False
                break
            used_edges.add(first_key)
            used_edges.add(second_key)
            graph[match.piece_a].add(match.piece_b)
            graph[match.piece_b].add(match.piece_a)
        if not valid:
            continue
        visited = {0}
        stack = [0]
        while stack:
            current = stack.pop()
            for neighbour in graph[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        if len(visited) != piece_count:
            continue
        yield combination
        emitted += 1
        if emitted >= maximum_layouts:
            return


def _assemble(
    pieces: Sequence[Piece], matches: Sequence[EdgeMatch]
) -> tuple[Transform, ...]:
    adjacency: list[list[tuple[int, int, int]]] = [
        [] for _ in range(len(pieces))
    ]
    for match in matches:
        adjacency[match.piece_a].append(
            (match.piece_b, match.edge_a, match.edge_b)
        )
        adjacency[match.piece_b].append(
            (match.piece_a, match.edge_b, match.edge_a)
        )
    transforms: list[Transform | None] = [None] * len(pieces)
    transforms[0] = identity_transform()
    stack = [0]
    while stack:
        current = stack.pop()
        current_transform = transforms[current]
        if current_transform is None:
            continue
        for neighbour, current_edge_index, neighbour_edge_index in adjacency[
            current
        ]:
            current_edge = polygon_edges(pieces[current].vertices)[
                current_edge_index
            ]
            neighbour_edge = polygon_edges(pieces[neighbour].vertices)[
                neighbour_edge_index
            ]
            world_a = apply_transform(current_edge[0], current_transform)
            world_b = apply_transform(current_edge[1], current_transform)
            proposed = _align_edge(
                neighbour_edge[0],
                neighbour_edge[1],
                world_b,
                world_a,
            )
            if transforms[neighbour] is None:
                transforms[neighbour] = proposed
                stack.append(neighbour)
    if any(transform is None for transform in transforms):
        raise PuzzleSolveError("匹配边没有形成连通拼图")
    return tuple(transform for transform in transforms if transform is not None)


def _point_in_triangle(point: Point, triangle: Sequence[Point]) -> bool:
    first = _cross(triangle[0], triangle[1], point)
    second = _cross(triangle[1], triangle[2], point)
    third = _cross(triangle[2], triangle[0], point)
    has_negative = min(first, second, third) < -_EPSILON
    has_positive = max(first, second, third) > _EPSILON
    return not (has_negative and has_positive)


def _triangulate(vertices: Sequence[Point]) -> list[tuple[Point, Point, Point]]:
    points = list(vertices)
    if signed_area(points) < 0:
        points.reverse()
    if len(points) == 3:
        return [(points[0], points[1], points[2])]
    indices = list(range(len(points)))
    triangles = []
    guard = 0
    while len(indices) > 3 and guard < 20:
        guard += 1
        ear_found = False
        for position, current_index in enumerate(indices):
            previous_index = indices[(position - 1) % len(indices)]
            next_index = indices[(position + 1) % len(indices)]
            triangle = (
                points[previous_index],
                points[current_index],
                points[next_index],
            )
            if _cross(*triangle) <= _EPSILON:
                continue
            if any(
                _point_in_triangle(points[other_index], triangle)
                for other_index in indices
                if other_index
                not in (previous_index, current_index, next_index)
            ):
                continue
            triangles.append(triangle)
            indices.pop(position)
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) == 3:
        triangles.append(
            (points[indices[0]], points[indices[1]], points[indices[2]])
        )
    if len(triangles) != len(points) - 2:
        # Camera pieces are normally convex; retain a deterministic fallback for
        # tiny fit noise that prevents the ear test from finding a strict ear.
        triangles = [
            (points[0], points[index], points[index + 1])
            for index in range(1, len(points) - 1)
        ]
    return triangles


def _line_intersection(
    segment_start: Point,
    segment_end: Point,
    line_start: Point,
    line_end: Point,
) -> Point:
    segment_direction = (
        segment_end[0] - segment_start[0],
        segment_end[1] - segment_start[1],
    )
    line_direction = (
        line_end[0] - line_start[0],
        line_end[1] - line_start[1],
    )
    denominator = (
        (segment_direction[0] * line_direction[1])
        - (segment_direction[1] * line_direction[0])
    )
    if abs(denominator) < _EPSILON:
        return segment_end
    offset = (
        line_start[0] - segment_start[0],
        line_start[1] - segment_start[1],
    )
    ratio = (
        (offset[0] * line_direction[1])
        - (offset[1] * line_direction[0])
    ) / denominator
    return (
        segment_start[0] + (ratio * segment_direction[0]),
        segment_start[1] + (ratio * segment_direction[1]),
    )


def _convex_intersection(
    subject: Sequence[Point], clip_polygon: Sequence[Point]
) -> list[Point]:
    output = list(subject)
    clip = list(clip_polygon)
    if signed_area(clip) < 0:
        clip.reverse()
    for edge_index, clip_start in enumerate(clip):
        clip_end = clip[(edge_index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _cross(clip_start, clip_end, previous) >= -_EPSILON
        for current in input_points:
            current_inside = (
                _cross(clip_start, clip_end, current) >= -_EPSILON
            )
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(
                            previous, current, clip_start, clip_end
                        )
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(
                        previous, current, clip_start, clip_end
                    )
                )
            previous = current
            previous_inside = current_inside
    return output


def polygon_overlap_area(
    first_polygon: Sequence[Point], second_polygon: Sequence[Point]
) -> float:
    total = 0.0
    for first_triangle in _triangulate(first_polygon):
        for second_triangle in _triangulate(second_polygon):
            intersection = _convex_intersection(
                first_triangle, second_triangle
            )
            if len(intersection) >= 3:
                total += polygon_area(intersection)
    return total


def _convex_hull(points: Sequence[Point]) -> list[Point]:
    unique = sorted(set((float(point[0]), float(point[1])) for point in points))
    if len(unique) <= 2:
        return unique
    lower = []
    for point in unique:
        while (
            len(lower) >= 2
            and _cross(lower[-2], lower[-1], point) <= _EPSILON
        ):
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while (
            len(upper) >= 2
            and _cross(upper[-2], upper[-1], point) <= _EPSILON
        ):
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _bounds(points: Sequence[Point]) -> tuple[float, float, float, float]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _minimum_area_axis(points: Sequence[Point]) -> tuple[float, float, float]:
    hull = _convex_hull(points)
    if len(hull) < 2:
        raise PuzzleSolveError("无法计算拼图外接矩形")
    best = None
    for index, first in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        rotated = transform_polygon(points, rigid_transform(-angle))
        minimum_x, minimum_y, maximum_x, maximum_y = _bounds(rotated)
        width = maximum_x - minimum_x
        height = maximum_y - minimum_y
        candidate = (width * height, angle, width, height)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise PuzzleSolveError("无法计算拼图方向")
    _, axis_angle, width, height = best
    if width < height:
        axis_angle += math.pi / 2.0
        width, height = height, width
    # A rectangle axis is equivalent after 180 degrees. Choose the smallest
    # rotation so animation does not add an unnecessary half turn.
    while axis_angle >= math.pi / 2.0:
        axis_angle -= math.pi
    while axis_angle < -math.pi / 2.0:
        axis_angle += math.pi
    return axis_angle, width, height


def _layout_score(
    pieces: Sequence[Piece],
    transforms: Sequence[Transform],
    matches: Sequence[EdgeMatch],
) -> float:
    assembled = [
        transform_polygon(piece.vertices, transform)
        for piece, transform in zip(pieces, transforms)
    ]
    total_area = sum(polygon_area(polygon) for polygon in assembled)
    overlap = 0.0
    for first_index, second_index in itertools.combinations(
        range(len(assembled)), 2
    ):
        overlap += polygon_overlap_area(
            assembled[first_index], assembled[second_index]
        )
    all_points = [point for polygon in assembled for point in polygon]
    _, rectangle_width, rectangle_height = _minimum_area_axis(all_points)
    rectangle_area = rectangle_width * rectangle_height
    union_area = max(0.0, total_area - overlap)
    fill_error = max(0.0, rectangle_area - union_area)
    length_error = sum(match.relative_error for match in matches)
    return (
        (overlap * 10.0)
        + (fill_error * 2.0)
        + (length_error * total_area * 0.35)
    )


def _normalise_to_target(
    pieces: Sequence[Piece],
    assembled_transforms: Sequence[Transform],
    frame_size: tuple[int, int],
    divider_y: float,
    target_margin: float,
) -> tuple[tuple[Transform, ...], tuple[float, float, float, float]]:
    assembled = [
        transform_polygon(piece.vertices, transform)
        for piece, transform in zip(pieces, assembled_transforms)
    ]
    all_points = [point for polygon in assembled for point in polygon]
    axis_angle, _, _ = _minimum_area_axis(all_points)
    normaliser = rigid_transform(-axis_angle)
    rotated_points = transform_polygon(all_points, normaliser)
    minimum_x, minimum_y, maximum_x, maximum_y = _bounds(rotated_points)
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    frame_width, frame_height = frame_size
    available_width = frame_width - (2.0 * target_margin)
    available_height = frame_height - divider_y - (2.0 * target_margin)
    if width > available_width or height > available_height:
        raise PuzzleSolveError(
            "拼好后的矩形 %.1f×%.1f px 放不进下半区 %.1f×%.1f px"
            % (width, height, available_width, available_height)
        )
    target_x = (frame_width - width) / 2.0
    target_y = divider_y + ((frame_height - divider_y - height) / 2.0)
    placement = rigid_transform(
        0.0, target_x - minimum_x, target_y - minimum_y
    )
    final_transforms = tuple(
        compose_transform(
            placement, compose_transform(normaliser, assembled_transform)
        )
        for assembled_transform in assembled_transforms
    )
    return final_transforms, (target_x, target_y, width, height)


def _mean_point(points: Sequence[Point]) -> Point:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _similarity_fit(
    source_points: Sequence[Point], target_points: Sequence[Point]
) -> tuple[float, float]:
    """Return optimal uniform scale and scale-normalised RMS fit error."""
    source_center = _mean_point(source_points)
    target_center = _mean_point(target_points)
    source_centered = [
        (
            point[0] - source_center[0],
            point[1] - source_center[1],
        )
        for point in source_points
    ]
    target_centered = [
        (
            point[0] - target_center[0],
            point[1] - target_center[1],
        )
        for point in target_points
    ]
    denominator = sum(
        (point[0] * point[0]) + (point[1] * point[1])
        for point in source_centered
    )
    target_radius_squared = sum(
        (point[0] * point[0]) + (point[1] * point[1])
        for point in target_centered
    ) / len(target_centered)
    if denominator < _EPSILON or target_radius_squared < _EPSILON:
        raise PuzzleSolveError("固定模板或检测轮廓发生退化")
    cosine_term = sum(
        (source[0] * target[0]) + (source[1] * target[1])
        for source, target in zip(source_centered, target_centered)
    )
    sine_term = sum(
        (source[0] * target[1]) - (source[1] * target[0])
        for source, target in zip(source_centered, target_centered)
    )
    magnitude = math.hypot(cosine_term, sine_term)
    scale = magnitude / denominator
    angle = math.atan2(sine_term, cosine_term)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    squared_error = 0.0
    for source, target in zip(source_centered, target_centered):
        mapped_x = scale * ((cosine * source[0]) - (sine * source[1]))
        mapped_y = scale * ((sine * source[0]) + (cosine * source[1]))
        squared_error += (
            ((mapped_x - target[0]) ** 2)
            + ((mapped_y - target[1]) ** 2)
        )
    rms_error = math.sqrt(squared_error / len(source_points))
    normalized_error = rms_error / math.sqrt(target_radius_squared)
    return scale, normalized_error


def _rigid_fit(
    source_points: Sequence[Point], target_points: Sequence[Point]
) -> Transform:
    """Return the least-squares rigid transform from source to target."""
    source_center = _mean_point(source_points)
    target_center = _mean_point(target_points)
    cosine_term = 0.0
    sine_term = 0.0
    for source, target in zip(source_points, target_points):
        source_x = source[0] - source_center[0]
        source_y = source[1] - source_center[1]
        target_x = target[0] - target_center[0]
        target_y = target[1] - target_center[1]
        cosine_term += (source_x * target_x) + (source_y * target_y)
        sine_term += (source_x * target_y) - (source_y * target_x)
    angle = math.atan2(sine_term, cosine_term)
    rotation = rigid_transform(angle)
    rotated_center = apply_transform(source_center, rotation)
    a, b, c, d, _, _ = rotation
    return (
        a,
        b,
        c,
        d,
        target_center[0] - rotated_center[0],
        target_center[1] - rotated_center[1],
    )


def _best_fixed_template_fit(
    piece: Piece, template_index: int
) -> _TemplateFit | None:
    template = QUESTION_ONE_TEMPLATES[template_index]
    if len(piece.vertices) != len(template.vertices_cm):
        return None
    best = None
    vertex_count = len(template.vertices_cm)
    for shift in range(vertex_count):
        aligned = tuple(
            template.vertices_cm[(index + shift) % vertex_count]
            for index in range(vertex_count)
        )
        pixels_per_cm, normalized_error = _similarity_fit(
            aligned, piece.vertices
        )
        fit = _TemplateFit(
            template_index,
            aligned,
            shift,
            pixels_per_cm,
            normalized_error,
        )
        if best is None or fit.normalized_error < best.normalized_error:
            best = fit
    return best


def solve_question_one_fixed(
    pieces: Sequence[Piece],
    frame_size: tuple[int, int] = (640, 480),
    divider_y: float | None = None,
    maximum_shape_error: float | None = None,
    maximum_scale_spread: float | None = None,
    vertex_tolerance_cm: float = 2.0,
    clearance_cm: float = 0.5,
    target_margin: float = 8.0,
) -> PuzzleSolution:
    """Recognise and restore the four fixed pieces from PDF question 1.

    Identity matching is invariant to translation, in-plane rotation and
    uniform pixel scale.  A robust global px/cm scale is then inferred from all
    four known templates.  The reference geometry determines identity and
    relative placement only: returned pieces retain their camera-measured
    outlines and every animation transform is rigid.  A configurable clearance
    is added between the measured target polygons for robot placement.
    """
    pieces = tuple(pieces)
    if len(pieces) != 4:
        raise PuzzleSolveError(
            "第一题固定形状模式要求摄像头恰好识别到 4 块碎片"
        )
    if frame_size[0] <= 0 or frame_size[1] <= 0:
        raise PuzzleSolveError("画面尺寸必须为正数")
    if divider_y is None:
        divider_y = frame_size[1] / 2.0

    fit_table: list[list[_TemplateFit | None]] = []
    for piece in pieces:
        fit_table.append(
            [
                _best_fixed_template_fit(piece, template_index)
                for template_index in range(len(QUESTION_ONE_TEMPLATES))
            ]
        )

    best_assignment = None
    best_assignment_score = None
    for permutation in itertools.permutations(
        range(len(QUESTION_ONE_TEMPLATES))
    ):
        selected = [
            fit_table[piece_index][template_index]
            for piece_index, template_index in enumerate(permutation)
        ]
        if any(fit is None for fit in selected):
            continue
        valid_fits = [fit for fit in selected if fit is not None]
        scales = [fit.pixels_per_cm for fit in valid_fits]
        median_scale = statistics.median(scales)
        scale_variance = sum(
            ((scale / median_scale) - 1.0) ** 2 for scale in scales
        ) / len(scales)
        shape_score = sum(
            fit.normalized_error**2 for fit in valid_fits
        )
        # The pieces have fixed physical sizes, so agreement on px/cm is strong
        # evidence of identity when one or two camera vertices are inaccurate.
        # This weight deliberately lets scale consistency overrule a slightly
        # better but physically impossible outline-only permutation.
        assignment_score = shape_score + (0.80 * scale_variance)
        if (
            best_assignment_score is None
            or assignment_score < best_assignment_score
        ):
            best_assignment_score = assignment_score
            best_assignment = tuple(valid_fits)
    if best_assignment is None or best_assignment_score is None:
        detected_sides = ", ".join(
            "P%d=%d边" % (piece.piece_id, len(piece.vertices))
            for piece in pieces
        )
        raise PuzzleSolveError(
            "4块轮廓无法与第一题的1个三角形和3个四边形对应："
            + detected_sides
        )

    errors = [fit.normalized_error for fit in best_assignment]
    if (
        maximum_shape_error is not None
        and max(errors) > maximum_shape_error
    ):
        details = ", ".join(
            "P%d=%.1f%%" % (piece.piece_id, error * 100.0)
            for piece, error in zip(pieces, errors)
        )
        raise PuzzleSolveError(
            "检测轮廓与第一题固定模板差异过大：" + details
        )
    scales = [fit.pixels_per_cm for fit in best_assignment]
    pixels_per_cm = float(statistics.median(scales))
    scale_spread = (max(scales) - min(scales)) / pixels_per_cm
    if (
        maximum_scale_spread is not None
        and scale_spread > maximum_scale_spread
    ):
        raise PuzzleSolveError(
            "4块碎片推算的像素比例不一致：%.1f%%，请检查透视或顶点识别"
            % (scale_spread * 100.0)
        )
    if vertex_tolerance_cm <= 0.0:
        raise PuzzleSolveError("顶点容差必须大于 0 cm")
    if not 0.0 <= clearance_cm <= 2.0:
        raise PuzzleSolveError("第一题预留间隙必须在 0～2 cm 之间")
    warnings = []

    frame_width, frame_height = frame_size
    target_width = 10.0 * pixels_per_cm
    target_height = 6.0 * pixels_per_cm
    available_width = frame_width - (2.0 * target_margin)
    available_height = frame_height - float(divider_y) - (
        2.0 * target_margin
    )
    if target_width > available_width or target_height > available_height:
        warnings.append(
            "参考尺寸按 %.2f px/cm 为 %.1f×%.1f px，无法完全放入"
            "下半区；不会缩放实际轮廓"
            % (pixels_per_cm, target_width, target_height)
        )
    target_x = (frame_width - target_width) / 2.0
    target_y = float(divider_y) + (
        (frame_height - float(divider_y) - target_height) / 2.0
    )
    target_pose = rigid_transform(0.0, target_x, target_y)

    base_transforms = []
    base_target_polygons = []
    reference_centers = []
    template_matches = []
    for piece, fit in zip(pieces, best_assignment):
        template = QUESTION_ONE_TEMPLATES[fit.template_index]
        aligned_template_px = tuple(
            (
                point[0] * pixels_per_cm,
                point[1] * pixels_per_cm,
            )
            for point in fit.aligned_vertices_cm
        )
        current_pose = _rigid_fit(aligned_template_px, piece.vertices)
        transform = compose_transform(
            target_pose,
            inverse_rigid_transform(current_pose),
        )
        base_transforms.append(transform)
        base_target_polygons.append(
            tuple(transform_polygon(piece.vertices, transform))
        )
        template_center = polygon_centroid(template.vertices_cm)
        reference_centers.append(
            (
                target_x + (template_center[0] * pixels_per_cm),
                target_y + (template_center[1] * pixels_per_cm),
            )
        )

        residuals_cm = [
            distance(
                apply_transform(template_point, current_pose),
                observed_point,
            )
            / pixels_per_cm
            for template_point, observed_point in zip(
                aligned_template_px, piece.vertices
            )
        ]
        observation_rms_cm = math.sqrt(
            sum(residual**2 for residual in residuals_cm)
            / len(residuals_cm)
        )
        observation_max_cm = max(residuals_cm)
        if observation_max_cm <= vertex_tolerance_cm * 0.5:
            confidence = "高"
        elif observation_max_cm <= vertex_tolerance_cm:
            confidence = "中"
        else:
            confidence = "低"
            warnings.append(
                "P%d/%s 最大顶点偏差 %.2f cm 超过 %.2f cm；"
                "仍保留实际轮廓并按最佳参考姿态归位"
                % (
                    piece.piece_id,
                    template.template_id,
                    observation_max_cm,
                    vertex_tolerance_cm,
                )
            )
        template_matches.append(
            FixedTemplateMatch(
                piece.piece_id,
                template.template_id,
                template.name,
                fit.normalized_error,
                fit.pixels_per_cm,
                fit.vertex_shift,
                confidence,
                observation_rms_cm,
                observation_max_cm,
            )
        )

    clearance_px = clearance_cm * pixels_per_cm
    piece_offsets = _separate_target_polygons(
        base_target_polygons,
        reference_centers,
        clearance_px,
    )
    final_transforms = [
        compose_transform(
            rigid_transform(0.0, offset[0], offset[1]),
            transform,
        )
        for transform, offset in zip(base_transforms, piece_offsets)
    ]
    target_polygons = [
        tuple(transform_polygon(piece.vertices, transform))
        for piece, transform in zip(pieces, final_transforms)
    ]
    unresolved_pairs = []
    for first_index in range(len(target_polygons)):
        for second_index in range(first_index + 1, len(target_polygons)):
            reference_direction = (
                reference_centers[second_index][0]
                - reference_centers[first_index][0],
                reference_centers[second_index][1]
                - reference_centers[first_index][1],
            )
            if _clearance_translation(
                target_polygons[first_index],
                target_polygons[second_index],
                reference_direction,
                clearance_px * 0.99,
            ) is not None:
                unresolved_pairs.append(
                    "P%d-P%d"
                    % (
                        pieces[first_index].piece_id,
                        pieces[second_index].piece_id,
                    )
                )
    if unresolved_pairs:
        warnings.append(
            "以下实际轮廓无法完全达到请求间隙："
            + "、".join(unresolved_pairs)
        )

    all_target_points = [
        point for polygon in target_polygons for point in polygon
    ]
    target_min_x = min(point[0] for point in all_target_points)
    target_max_x = max(point[0] for point in all_target_points)
    target_min_y = min(point[1] for point in all_target_points)
    target_max_y = max(point[1] for point in all_target_points)
    actual_target_width = target_max_x - target_min_x
    actual_target_height = target_max_y - target_min_y
    desired_min_x = (frame_width - actual_target_width) / 2.0
    if actual_target_height <= available_height:
        desired_min_y = float(divider_y) + target_margin + (
            (available_height - actual_target_height) / 2.0
        )
    elif actual_target_height <= frame_height - (2.0 * target_margin):
        # Preserve the measured size and keep the whole target visible, even
        # when noisy outlines plus clearance cannot stay wholly below divider.
        desired_min_y = (
            frame_height - target_margin - actual_target_height
        )
    else:
        desired_min_y = (frame_height - actual_target_height) / 2.0
    recenter_x = desired_min_x - target_min_x
    recenter_y = desired_min_y - target_min_y
    recenter = rigid_transform(0.0, recenter_x, recenter_y)
    final_transforms = [
        compose_transform(recenter, transform)
        for transform in final_transforms
    ]
    target_polygons = [
        tuple(transform_polygon(piece.vertices, transform))
        for piece, transform in zip(pieces, final_transforms)
    ]
    target_min_x += recenter_x
    target_max_x += recenter_x
    target_min_y += recenter_y
    target_max_y += recenter_y
    if (
        actual_target_width > available_width
        or actual_target_height > available_height
    ):
        warnings.append(
            "实际轮廓加 %.2f cm 留隙后占用 %.1f×%.1f px，"
            "超过目标区可用 %.1f×%.1f px；已保持实际尺寸并居中显示"
            % (
                clearance_cm,
                actual_target_width,
                actual_target_height,
                available_width,
                available_height,
            )
        )

    motions = []
    for piece, transform in zip(pieces, final_transforms):
        source_center = polygon_centroid(piece.vertices)
        target_center = apply_transform(source_center, transform)
        angle_degrees = math.degrees(transform_angle(transform))
        while angle_degrees >= 180.0:
            angle_degrees -= 360.0
        while angle_degrees < -180.0:
            angle_degrees += 360.0
        motions.append(
            MotionPlan(
                piece.piece_id,
                source_center,
                target_center,
                angle_degrees,
                transform,
            )
        )

    motions.sort(
        key=lambda motion: (
            -motion.target_center[1],
            motion.target_center[0],
        )
    )
    return PuzzleSolution(
        pieces=pieces,
        transforms=tuple(final_transforms),
        matches=(),
        motions=tuple(motions),
        target_bounds=(
            target_min_x,
            target_min_y,
            actual_target_width,
            actual_target_height,
        ),
        score=float(best_assignment_score),
        mode="fixed_question_one",
        template_matches=tuple(template_matches),
        pixels_per_cm=pixels_per_cm,
        target_polygons=tuple(target_polygons),
        warnings=tuple(warnings),
        clearance_cm=clearance_cm,
    )


def solve_puzzle(
    pieces: Sequence[Piece],
    frame_size: tuple[int, int] = (640, 480),
    divider_y: float | None = None,
    length_tolerance: float = 0.14,
    maximum_candidates: int = 30,
    maximum_layouts: int = 6000,
    target_margin: float = 8.0,
) -> PuzzleSolution:
    """Assemble camera polygons and place the result in the lower frame half."""
    pieces = tuple(pieces)
    if not 1 <= len(pieces) <= 4:
        raise PuzzleSolveError("拼图数量必须为 1～4 块")
    if frame_size[0] <= 0 or frame_size[1] <= 0:
        raise PuzzleSolveError("画面尺寸必须为正数")
    if divider_y is None:
        divider_y = frame_size[1] / 2.0

    if len(pieces) == 1:
        best_transforms = (identity_transform(),)
        best_matches: tuple[EdgeMatch, ...] = ()
        best_score = 0.0
    else:
        candidates = _candidate_matches(
            pieces, length_tolerance, maximum_candidates
        )
        if not candidates:
            raise PuzzleSolveError("没有找到长度相近的可拼接边")
        best_transforms = None
        best_matches = None
        best_score = None
        for matches in _connected_matching_sets(
            candidates, len(pieces), maximum_layouts
        ):
            transforms = _assemble(pieces, matches)
            score = _layout_score(pieces, transforms, matches)
            if best_score is None or score < best_score:
                best_score = score
                best_transforms = transforms
                best_matches = tuple(matches)
        if best_transforms is None or best_matches is None or best_score is None:
            raise PuzzleSolveError("没有找到连通且接近矩形的拼接方案")

    final_transforms, target_bounds = _normalise_to_target(
        pieces,
        best_transforms,
        frame_size,
        float(divider_y),
        target_margin,
    )
    motions = []
    for piece, transform in zip(pieces, final_transforms):
        source_center = polygon_centroid(piece.vertices)
        target_center = apply_transform(source_center, transform)
        angle_degrees = math.degrees(transform_angle(transform))
        while angle_degrees >= 180.0:
            angle_degrees -= 360.0
        while angle_degrees < -180.0:
            angle_degrees += 360.0
        motions.append(
            MotionPlan(
                piece.piece_id,
                source_center,
                target_center,
                angle_degrees,
                transform,
            )
        )
    # Place the pieces furthest into the target first. This gives the animation
    # and a real top-down mechanism a simple, repeatable placement order.
    motions.sort(
        key=lambda motion: (
            -motion.target_center[1],
            motion.target_center[0],
        )
    )
    return PuzzleSolution(
        pieces,
        final_transforms,
        tuple(best_matches),
        tuple(motions),
        target_bounds,
        float(best_score),
    )


def interpolate_polygon(
    vertices: Sequence[Point],
    transform: Transform,
    fraction: float,
) -> list[Point]:
    """Interpolate a rigid move around the piece centroid for animation."""
    fraction = max(0.0, min(1.0, float(fraction)))
    source_center = polygon_centroid(vertices)
    target_center = apply_transform(source_center, transform)
    current_center = (
        source_center[0]
        + ((target_center[0] - source_center[0]) * fraction),
        source_center[1]
        + ((target_center[1] - source_center[1]) * fraction),
    )
    rotation = rigid_transform(transform_angle(transform) * fraction)
    result = []
    for point in vertices:
        local_point = (
            point[0] - source_center[0],
            point[1] - source_center[1],
        )
        rotated = apply_transform(local_point, rotation)
        result.append(
            (
                rotated[0] + current_center[0],
                rotated[1] + current_center[1],
            )
        )
    return result


def transform_as_matrix(transform: Transform) -> list[list[float]]:
    a, b, c, d, tx, ty = transform
    return [[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]]
