"""Pure-Python polygon geometry helpers for CanMV K230.

The module deliberately avoids CPython-only modules so the same code can run
under CanMV MicroPython and in host-side tests.
"""

import math


_EPSILON = 1e-9


def distance(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return math.sqrt((dx * dx) + (dy * dy))


def signed_area(vertices):
    total = 0.0
    count = len(vertices)
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        total += (x1 * y2) - (y1 * x2)
    return total * 0.5


def centroid(vertices):
    """Return an area centroid, with an arithmetic-mean fallback."""
    area_twice = 0.0
    x_total = 0.0
    y_total = 0.0
    count = len(vertices)

    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        cross = (x1 * y2) - (x2 * y1)
        area_twice += cross
        x_total += (x1 + x2) * cross
        y_total += (y1 + y2) * cross

    if abs(area_twice) < _EPSILON:
        return (
            sum(point[0] for point in vertices) / count,
            sum(point[1] for point in vertices) / count,
        )

    scale = 1.0 / (3.0 * area_twice)
    return x_total * scale, y_total * scale


def _point_line_distance(point, line_start, line_end):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    denominator = math.sqrt((dx * dx) + (dy * dy))
    if denominator < _EPSILON:
        return distance(point, line_start)
    numerator = abs(
        (dy * point[0])
        - (dx * point[1])
        + (line_end[0] * line_start[1])
        - (line_end[1] * line_start[0])
    )
    return numerator / denominator


def _deduplicate_contour(points, minimum_spacing=0.5):
    if not points:
        return []

    result = [(float(points[0][0]), float(points[0][1]))]
    for point in points[1:]:
        converted = (float(point[0]), float(point[1]))
        if distance(result[-1], converted) >= minimum_spacing:
            result.append(converted)

    if len(result) > 1 and distance(result[0], result[-1]) < minimum_spacing:
        result.pop()
    return result


def _cyclic_indices(start, end, count):
    indices = [start]
    index = start
    while index != end:
        index = (index + 1) % count
        indices.append(index)
    return indices


def _rdp_open_indices(points, chain, epsilon):
    """Iterative Ramer-Douglas-Peucker over a chain of contour indices."""
    keep = [False] * len(chain)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(chain) - 1)]

    while stack:
        start_position, end_position = stack.pop()
        start_point = points[chain[start_position]]
        end_point = points[chain[end_position]]
        farthest_position = -1
        farthest_distance = -1.0

        for position in range(start_position + 1, end_position):
            candidate_distance = _point_line_distance(
                points[chain[position]], start_point, end_point
            )
            if candidate_distance > farthest_distance:
                farthest_distance = candidate_distance
                farthest_position = position

        if farthest_position >= 0 and farthest_distance > epsilon:
            keep[farthest_position] = True
            stack.append((start_position, farthest_position))
            stack.append((farthest_position, end_position))

    return [chain[index] for index, should_keep in enumerate(keep) if should_keep]


def simplify_closed_contour_indices(contour, epsilon):
    """Return contour indices for a closed RDP polygon approximation.

    A closed contour has no natural first/last point. Splitting at an
    approximate diameter keeps the arbitrary trace seam from hiding a corner.
    """
    points = _deduplicate_contour(contour)
    count = len(points)
    if count < 3:
        return points, list(range(count))

    anchor_a = max(range(count), key=lambda i: distance(points[0], points[i]))
    anchor_b = max(
        range(count), key=lambda i: distance(points[anchor_a], points[i])
    )

    if anchor_a == anchor_b:
        return points, list(range(count))

    chain_a = _cyclic_indices(anchor_a, anchor_b, count)
    chain_b = _cyclic_indices(anchor_b, anchor_a, count)
    simplified_a = _rdp_open_indices(points, chain_a, epsilon)
    simplified_b = _rdp_open_indices(points, chain_b, epsilon)

    # Both chains contain the two anchors. Keep each anchor exactly once.
    combined = simplified_a + simplified_b[1:-1]
    return points, combined


def _edge_samples(points, start_index, end_index):
    indices = _cyclic_indices(start_index, end_index, len(points))
    return [points[index] for index in indices]


def _fit_line(points):
    """Fit ax + by + c = 0 using total least squares."""
    count = len(points)
    if count < 2:
        return None

    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    xx = 0.0
    yy = 0.0
    xy = 0.0

    for x, y in points:
        dx = x - mean_x
        dy = y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy

    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    normal_a = -direction_y
    normal_b = direction_x
    normal_c = -((normal_a * mean_x) + (normal_b * mean_y))
    return normal_a, normal_b, normal_c


def _intersect_lines(line_a, line_b):
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    determinant = (a1 * b2) - (a2 * b1)
    if abs(determinant) < _EPSILON:
        return None

    x = ((b1 * c2) - (b2 * c1)) / determinant
    y = ((c1 * a2) - (c2 * a1)) / determinant
    return x, y


def refine_polygon(contour, vertex_indices, maximum_shift=12.0):
    """Fit each polygon side and intersect neighbouring fitted lines."""
    if len(vertex_indices) < 3:
        return []

    lines = []
    for index in range(len(vertex_indices)):
        samples = _edge_samples(
            contour,
            vertex_indices[index],
            vertex_indices[(index + 1) % len(vertex_indices)],
        )
        lines.append(_fit_line(samples))

    refined = []
    for index in range(len(vertex_indices)):
        original = contour[vertex_indices[index]]
        previous_line = lines[index - 1]
        next_line = lines[index]
        intersection = None
        if previous_line is not None and next_line is not None:
            intersection = _intersect_lines(previous_line, next_line)

        if (
            intersection is None
            or distance(original, intersection) > maximum_shift
        ):
            refined.append(original)
        else:
            refined.append(intersection)
    return refined


def _prune_short_corner_pairs(contour, vertex_indices, minimum_edge_length):
    """Remove RDP corner duplicates caused by one-pixel boundary stair-steps."""
    indices = list(vertex_indices)
    while len(indices) > 5:
        shortest_position = -1
        shortest_length = None
        for position in range(len(indices)):
            edge_length = distance(
                contour[indices[position]],
                contour[indices[(position + 1) % len(indices)]],
            )
            if shortest_length is None or edge_length < shortest_length:
                shortest_length = edge_length
                shortest_position = position

        if shortest_length is None or shortest_length >= minimum_edge_length:
            break

        first_position = shortest_position
        second_position = (shortest_position + 1) % len(indices)
        previous_position = (first_position - 1) % len(indices)
        next_position = (second_position + 1) % len(indices)

        remove_first_error = _point_line_distance(
            contour[indices[first_position]],
            contour[indices[previous_position]],
            contour[indices[second_position]],
        )
        remove_second_error = _point_line_distance(
            contour[indices[second_position]],
            contour[indices[first_position]],
            contour[indices[next_position]],
        )
        remove_position = (
            first_position
            if remove_first_error <= remove_second_error
            else second_position
        )
        indices.pop(remove_position)
    return indices


def normalize_vertices(vertices):
    """Return clockwise image-coordinate vertices with a stable first point."""
    normalized = [(float(point[0]), float(point[1])) for point in vertices]
    if len(normalized) < 3:
        return normalized

    # With image Y increasing downwards, a positive shoelace area is clockwise.
    if signed_area(normalized) < 0:
        normalized.reverse()

    first_index = min(
        range(len(normalized)),
        key=lambda index: (
            normalized[index][0] + normalized[index][1],
            normalized[index][1],
            normalized[index][0],
        ),
    )
    return normalized[first_index:] + normalized[:first_index]


def _orientation(point_a, point_b, point_c):
    return (
        ((point_b[0] - point_a[0]) * (point_c[1] - point_a[1]))
        - ((point_b[1] - point_a[1]) * (point_c[0] - point_a[0]))
    )


def _segments_intersect(a1, a2, b1, b2):
    first = _orientation(a1, a2, b1)
    second = _orientation(a1, a2, b2)
    third = _orientation(b1, b2, a1)
    fourth = _orientation(b1, b2, a2)
    return (
        ((first > _EPSILON and second < -_EPSILON)
         or (first < -_EPSILON and second > _EPSILON))
        and
        ((third > _EPSILON and fourth < -_EPSILON)
         or (third < -_EPSILON and fourth > _EPSILON))
    )


def is_self_intersecting(vertices):
    count = len(vertices)
    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
            ):
                continue
            if _segments_intersect(
                vertices[first_index],
                vertices[first_next],
                vertices[second_index],
                vertices[second_next],
            ):
                return True
    return False


def validate_polygon(
    vertices,
    minimum_vertices=3,
    maximum_vertices=5,
    minimum_edge_length=0.0,
    minimum_area=1.0,
):
    if len(vertices) < minimum_vertices:
        return False, "too_few_vertices"
    if len(vertices) > maximum_vertices:
        return False, "too_many_vertices"
    if abs(signed_area(vertices)) < minimum_area:
        return False, "area_too_small"
    if is_self_intersecting(vertices):
        return False, "self_intersection"

    for index in range(len(vertices)):
        if (
            distance(vertices[index], vertices[(index + 1) % len(vertices)])
            < minimum_edge_length
        ):
            return False, "edge_too_short"
    return True, None


def polygon_from_contour(
    contour,
    rdp_epsilon,
    minimum_edge_length,
    minimum_area,
    maximum_vertex_shift=None,
):
    """Approximate, refine, normalize and validate a traced contour."""
    cleaned, indices = simplify_closed_contour_indices(contour, rdp_epsilon)
    if len(indices) > 5:
        indices = _prune_short_corner_pairs(
            cleaned, indices, minimum_edge_length
        )
    if maximum_vertex_shift is None:
        maximum_vertex_shift = max(8.0, rdp_epsilon * 4.0)

    if len(indices) < 3 or len(indices) > 5:
        rough = [cleaned[index] for index in indices]
        return normalize_vertices(rough), (
            "too_few_vertices" if len(indices) < 3 else "too_many_vertices"
        )

    refined = refine_polygon(cleaned, indices, maximum_vertex_shift)
    vertices = normalize_vertices(refined)
    valid, reason = validate_polygon(
        vertices,
        minimum_vertices=3,
        maximum_vertices=5,
        minimum_edge_length=minimum_edge_length,
        minimum_area=minimum_area,
    )
    return vertices, reason if not valid else None


def measure_polygon(vertices, pixels_per_cm=None):
    edge_lengths_px = []
    for index in range(len(vertices)):
        edge_lengths_px.append(
            distance(vertices[index], vertices[(index + 1) % len(vertices)])
        )

    vertices_px = [
        [int(round(point[0])), int(round(point[1]))] for point in vertices
    ]
    edge_lengths_px_rounded = [round(value, 2) for value in edge_lengths_px]
    perimeter_px = round(sum(edge_lengths_px), 2)

    if pixels_per_cm is None or pixels_per_cm <= 0:
        vertices_cm = None
        edge_lengths_cm = None
        perimeter_cm = None
    else:
        vertices_cm = [
            [
                round(point[0] / pixels_per_cm, 3),
                round(point[1] / pixels_per_cm, 3),
            ]
            for point in vertices
        ]
        edge_lengths_cm = [
            round(value / pixels_per_cm, 3) for value in edge_lengths_px
        ]
        perimeter_cm = round(sum(edge_lengths_px) / pixels_per_cm, 3)

    center = centroid(vertices)
    return {
        "vertices_px": vertices_px,
        "vertices_cm": vertices_cm,
        "edge_lengths_px": edge_lengths_px_rounded,
        "edge_lengths_cm": edge_lengths_cm,
        "perimeter_px": perimeter_px,
        "perimeter_cm": perimeter_cm,
        "centroid_px": [int(round(center[0])), int(round(center[1]))],
    }
