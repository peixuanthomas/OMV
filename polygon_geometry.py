"""Pure-Python polygon geometry helpers for OpenMV.

The module deliberately avoids CPython-only modules so the same code can run
under OpenMV MicroPython and in host-side tests.
"""

import math


_EPSILON = 1e-9
# The physical pieces do not contain almost-straight "corners". A fitted
# vertex at or above this interior angle is threshold stair-stepping and the
# two adjacent segments should be treated as one edge.
COLLINEAR_MERGE_ANGLE_DEG = 170.0


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


def _interior_angle_degrees(previous, vertex, following):
    first_x = previous[0] - vertex[0]
    first_y = previous[1] - vertex[1]
    second_x = following[0] - vertex[0]
    second_y = following[1] - vertex[1]
    first_length = math.sqrt(
        (first_x * first_x) + (first_y * first_y)
    )
    second_length = math.sqrt(
        (second_x * second_x) + (second_y * second_y)
    )
    denominator = first_length * second_length
    if denominator < _EPSILON:
        return 0.0
    cosine = (
        (first_x * second_x) + (first_y * second_y)
    ) / denominator
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine) * (180.0 / math.pi)


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


def _fit_line_robust(points, maximum_residual=2.5):
    """Fit a line twice, discarding small contour bumps on the second pass."""
    line = _fit_line(points)
    if line is None or len(points) < 6:
        return line

    normal_a, normal_b, normal_c = line
    inliers = [
        point
        for point in points
        if abs(
            (normal_a * point[0])
            + (normal_b * point[1])
            + normal_c
        ) <= maximum_residual
    ]
    if len(inliers) < 4 or len(inliers) * 2 < len(points):
        return line
    return _fit_line(inliers)


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
        # RDP vertices sit on pixel stair-steps at each corner. Excluding a
        # small endpoint slice keeps those turns from rotating the fitted
        # straight side, while retaining enough pixels for short valid edges.
        trim = min(4, len(samples) // 12)
        if trim > 0 and (len(samples) - (trim * 2)) >= 4:
            samples = samples[trim:-trim]
        lines.append(_fit_line_robust(samples))

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
    # Three-to-five vertices are all legal output shapes.  Never turn one
    # legal shape into another merely because the *rough* RDP endpoints make
    # an edge look short: line fitting below can move those endpoints outward
    # and recover the true edge length.  Pruning is only safe when RDP has
    # produced more vertices than the detector can report.
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


def _refine_and_validate(
    contour,
    vertex_indices,
    epsilon,
    minimum_edge_length,
    minimum_area,
    maximum_vertex_shift,
):
    """Refine one RDP candidate and return its vertices and error reason."""
    if len(vertex_indices) < 3 or len(vertex_indices) > 5:
        rough = [contour[index] for index in vertex_indices]
        return normalize_vertices(rough), (
            "too_few_vertices"
            if len(vertex_indices) < 3
            else "too_many_vertices"
        )

    allowed_shift = maximum_vertex_shift
    if allowed_shift is None:
        allowed_shift = max(8.0, epsilon * 4.0)

    refined = refine_polygon(contour, vertex_indices, allowed_shift)
    vertices = normalize_vertices(refined)
    valid, reason = validate_polygon(
        vertices,
        minimum_vertices=3,
        maximum_vertices=5,
        minimum_edge_length=minimum_edge_length,
        minimum_area=minimum_area,
    )
    return vertices, reason if not valid else None


def _remove_invalid_short_edge_corner(
    contour,
    vertex_indices,
    epsilon,
    minimum_edge_length,
    maximum_vertex_shift,
):
    """Merge the less significant endpoint of one invalid fitted short edge."""
    indices = list(vertex_indices)
    if len(indices) <= 3:
        return indices

    allowed_shift = maximum_vertex_shift
    if allowed_shift is None:
        allowed_shift = max(8.0, epsilon * 4.0)
    refined = refine_polygon(contour, indices, allowed_shift)

    shortest_position = min(
        range(len(refined)),
        key=lambda position: distance(
            refined[position],
            refined[(position + 1) % len(refined)],
        ),
    )
    shortest_length = distance(
        refined[shortest_position],
        refined[(shortest_position + 1) % len(refined)],
    )
    if shortest_length >= minimum_edge_length:
        return indices

    first_position = shortest_position
    second_position = (shortest_position + 1) % len(indices)
    previous_position = (first_position - 1) % len(indices)
    next_position = (second_position + 1) % len(indices)
    remove_first_error = _point_line_distance(
        refined[first_position],
        refined[previous_position],
        refined[second_position],
    )
    remove_second_error = _point_line_distance(
        refined[second_position],
        refined[first_position],
        refined[next_position],
    )
    remove_position = (
        first_position
        if remove_first_error <= remove_second_error
        else second_position
    )
    indices.pop(remove_position)
    return indices


def _remove_refined_collinear_corner(
    contour,
    vertex_indices,
    epsilon,
    minimum_edge_length,
    maximum_vertex_shift,
):
    """Remove one long-edge split that becomes nearly collinear after fitting."""
    indices = list(vertex_indices)
    if len(indices) <= 3:
        return indices

    allowed_shift = maximum_vertex_shift
    if allowed_shift is None:
        allowed_shift = max(8.0, epsilon * 4.0)
    refined = refine_polygon(contour, indices, allowed_shift)
    minimum_supported_edge = minimum_edge_length
    best_position = None
    best_angle = None

    for position in range(len(refined)):
        previous_position = (position - 1) % len(refined)
        next_position = (position + 1) % len(refined)
        previous_length = distance(
            refined[previous_position], refined[position]
        )
        next_length = distance(
            refined[position], refined[next_position]
        )
        # Do not use a sub-minimum fragment as evidence for a corner. If both
        # incident sides are legal yet the fitted point is nearly collinear,
        # it is the unstable fifth-point split handled by this fallback.
        if (
            previous_length < minimum_supported_edge
            or next_length < minimum_supported_edge
        ):
            continue

        angle = _interior_angle_degrees(
            refined[previous_position],
            refined[position],
            refined[next_position],
        )
        if (
            angle >= COLLINEAR_MERGE_ANGLE_DEG
            and (best_angle is None or angle > best_angle)
        ):
            best_position = position
            best_angle = angle

    if best_position is not None:
        indices.pop(best_position)
    return indices


def _remove_collinear_fitted_vertex(
    vertices,
    minimum_edge_length,
):
    """Drop one nearly-collinear fitted point without refitting its neighbours."""
    reduced = list(vertices)
    if len(reduced) <= 3:
        return reduced

    best_position = None
    best_angle = None
    for position in range(len(reduced)):
        previous_position = (position - 1) % len(reduced)
        next_position = (position + 1) % len(reduced)
        if (
            distance(reduced[previous_position], reduced[position])
            < minimum_edge_length
            or distance(reduced[position], reduced[next_position])
            < minimum_edge_length
        ):
            continue
        angle = _interior_angle_degrees(
            reduced[previous_position],
            reduced[position],
            reduced[next_position],
        )
        if (
            angle >= COLLINEAR_MERGE_ANGLE_DEG
            and (best_angle is None or angle > best_angle)
        ):
            best_position = position
            best_angle = angle

    if best_position is not None:
        reduced.pop(best_position)
    return normalize_vertices(reduced)


def _merge_collinear_fitted_vertices(
    vertices,
    minimum_edge_length,
    minimum_area,
):
    """Merge every valid near-straight split, including a 4->3 cleanup."""
    merged = normalize_vertices(vertices)
    while len(merged) > 3:
        candidate = _remove_collinear_fitted_vertex(
            merged,
            minimum_edge_length,
        )
        if len(candidate) == len(merged):
            break

        valid, _ = validate_polygon(
            candidate,
            minimum_vertices=3,
            maximum_vertices=5,
            minimum_edge_length=minimum_edge_length,
            minimum_area=minimum_area,
        )
        if not valid:
            break
        merged = candidate
    return merged


def normalize_vertices(vertices):
    """Return clockwise image-coordinate vertices, starting at the top-left."""
    normalized = [(float(point[0]), float(point[1])) for point in vertices]
    if len(normalized) < 3:
        return normalized

    # With image Y increasing downwards, a positive shoelace area is clockwise.
    if signed_area(normalized) < 0:
        normalized.reverse()

    # Line intersections are sub-pixel values. On a nearly horizontal top
    # edge, tiny fit noise must not make V1 alternate between its two ends.
    top_y = min(point[1] for point in normalized)
    top_band = [
        index
        for index in range(len(normalized))
        if normalized[index][1] <= (top_y + 1.0)
    ]
    first_index = min(
        top_band,
        key=lambda index: (
            normalized[index][0],
            normalized[index][1],
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
    cleaned = _deduplicate_contour(contour)
    indices = []
    base_epsilon = float(rdp_epsilon)
    epsilon = base_epsilon

    # A one-pixel staircase or small reflection can create extra RDP corners.
    # The official pieces have at most five real sides, so progressively relax
    # the approximation until the result fits that physical constraint.
    for _ in range(5):
        cleaned, indices = simplify_closed_contour_indices(cleaned, epsilon)
        if len(indices) <= 5:
            break
        epsilon *= 1.35

    indices = _prune_short_corner_pairs(
        cleaned, indices, minimum_edge_length
    )
    vertices, reason = _refine_and_validate(
        cleaned,
        indices,
        epsilon,
        minimum_edge_length,
        minimum_area,
        maximum_vertex_shift,
    )

    # RDP can split a pointed or pixel-stair-stepped corner into two nearby
    # vertices. If the fitted result still has an edge below the physical
    # minimum, the candidate cannot legally keep both endpoints. Merge the
    # less significant endpoint and retry instead of rejecting the whole
    # piece. Refinement happens before this fallback, preserving genuine short
    # sides whose rough RDP endpoints initially underestimated their length.
    if reason == "edge_too_short" and len(indices) > 3:
        merged_indices = list(indices)
        while len(merged_indices) > 3:
            next_indices = _remove_invalid_short_edge_corner(
                cleaned,
                merged_indices,
                epsilon,
                minimum_edge_length,
                maximum_vertex_shift,
            )
            if len(next_indices) == len(merged_indices):
                break
            merged_indices = next_indices
            merged_vertices, merged_reason = _refine_and_validate(
                cleaned,
                merged_indices,
                epsilon,
                minimum_edge_length,
                minimum_area,
                maximum_vertex_shift,
            )
            if merged_reason is None:
                indices = merged_indices
                vertices = merged_vertices
                reason = None
                break
            if merged_reason != "edge_too_short":
                break

    # Robust line fitting can reveal that a noisy RDP corner merely split one
    # long physical side. Collapse only well-supported long-edge splits; this
    # leaves a genuine short side near the physical minimum untouched.
    if reason is None and len(indices) > 4:
        collinear_indices = list(indices)
        while len(collinear_indices) > 4:
            next_indices = _remove_refined_collinear_corner(
                cleaned,
                collinear_indices,
                epsilon,
                minimum_edge_length,
                maximum_vertex_shift,
            )
            if len(next_indices) == len(collinear_indices):
                break
            candidate_vertices, candidate_reason = _refine_and_validate(
                cleaned,
                next_indices,
                epsilon,
                minimum_edge_length,
                minimum_area,
                maximum_vertex_shift,
            )
            if candidate_reason is not None:
                break
            collinear_indices = next_indices
            indices = next_indices
            vertices = candidate_vertices

    # The same false corner can be the fourth point of a physical triangle.
    # The contour-index cleanup above deliberately targeted 5->4 candidates;
    # finish every successful fit in vertex space so a 170+ degree 4->3 split
    # cannot leak through either the primary or short-edge fallback path.
    if reason is None:
        vertices = _merge_collinear_fitted_vertices(
            vertices,
            minimum_edge_length,
            minimum_area,
        )

    # A high epsilon can bridge across a shallow but valid corner and make a
    # quadrilateral/pentagon look like a triangle. Retry that ambiguous case
    # at progressively lower epsilons. The former implementation stopped when
    # the first recovery scale still had three points, precisely when a still
    # lower epsilon was needed to reveal the fourth corner.
    if len(indices) <= 3 and base_epsilon > 1.5:
        recovery_epsilon = max(1.5, base_epsilon * 0.60)
        for _ in range(5):
            recovery_cleaned, recovery_indices = (
                simplify_closed_contour_indices(
                    cleaned, recovery_epsilon
                )
            )
            recovery_indices = _prune_short_corner_pairs(
                recovery_cleaned,
                recovery_indices,
                minimum_edge_length,
            )
            if 4 <= len(recovery_indices) <= 5:
                recovered_vertices, recovered_reason = (
                    _refine_and_validate(
                        recovery_cleaned,
                        recovery_indices,
                        recovery_epsilon,
                        minimum_edge_length,
                        minimum_area,
                        maximum_vertex_shift,
                    )
                )
                if recovered_reason is None:
                    # Recovery may expose both the missing shallow corner and
                    # one nearly-collinear staircase point on a long side.
                    # Apply the same conservative final cleanup used by the
                    # primary path before accepting the lower-epsilon result,
                    # including the 4->3 case for a split triangle side.
                    # Keep the already fitted vertices here: fitting the two
                    # joined contour spans again can pull a genuine short edge
                    # just below its validation tolerance.
                    recovered_vertices = (
                        _merge_collinear_fitted_vertices(
                            recovered_vertices,
                            minimum_edge_length,
                            minimum_area,
                        )
                    )
                    return recovered_vertices, None
            if recovery_epsilon <= 1.5:
                break
            recovery_epsilon = max(1.5, recovery_epsilon * 0.75)

    return vertices, reason


def contour_length(contour):
    """Return the length of a closed traced pixel chain."""
    if len(contour) < 2:
        return 0.0
    total = 0.0
    for index in range(len(contour)):
        total += distance(contour[index], contour[(index + 1) % len(contour)])
    return total


def measure_polygon(
    vertices,
    traced_boundary_length_px=None,
    edge_support=None,
):
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

    center = centroid(vertices)
    traced_boundary_length_px = (
        None
        if traced_boundary_length_px is None
        else round(traced_boundary_length_px, 2)
    )

    return {
        "vertex_count": len(vertices),
        "vertices_px": vertices_px,
        "edge_lengths_px": edge_lengths_px_rounded,
        "boundary_length_px": perimeter_px,
        "perimeter_px": perimeter_px,
        "traced_boundary_length_px": traced_boundary_length_px,
        "edge_support": (
            None if edge_support is None else round(edge_support, 3)
        ),
        "area_px2": round(abs(signed_area(vertices)), 2),
        "centroid_px": [int(round(center[0])), int(round(center[1]))],
    }
