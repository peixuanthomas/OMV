"""Host-side regression tests for the OpenMV polygon geometry helpers."""

import unittest

import polygon_geometry as geometry


def _raster_line(start, end):
    """Return integer pixels for one line, including both endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    step_x = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    pixels = []

    while True:
        pixels.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return pixels
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += step_x
        if doubled_error <= dx:
            error += dx
            y0 += step_y


def _raster_contour(vertices):
    contour = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        contour.extend(_raster_line(start, end)[:-1])
    return contour


class PolygonGeometryTests(unittest.TestCase):
    def test_recovers_valid_short_side_hidden_by_primary_rdp(self):
        # At epsilon 6 the 32.6 px side is bridged and the old code reported
        # this quadrilateral as a triangle.  At the recovery epsilon, robust
        # line fitting measures the side as approximately 32.9 px.
        contour = _raster_contour(
            [(306, 263), (134, 216), (291, 188), (301, 219)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(4, len(vertices))
        edge_lengths = [
            geometry.distance(
                vertices[index],
                vertices[(index + 1) % len(vertices)],
            )
            for index in range(len(vertices))
        ]
        self.assertGreaterEqual(min(edge_lengths), 32.0)

    def test_keeps_shortest_live_quadrilateral_side(self):
        # Fixed-camera regression extracted from the center piece: perspective
        # measures its shortest physical 20 px edge at about 19.25 px. The old
        # validation merged that edge and reported this quadrilateral as a
        # triangle. The detector now validates at 16 px to preserve it.
        contour = _raster_contour(
            [(210, 203), (353, 209), (356, 228), (286, 250)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=16.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(4, len(vertices))
        edge_lengths = [
            geometry.distance(
                vertices[index],
                vertices[(index + 1) % len(vertices)],
            )
            for index in range(len(vertices))
        ]
        self.assertGreaterEqual(min(edge_lengths), 16.0)
        self.assertLess(min(edge_lengths), 20.0)

    def test_short_rough_edge_is_refined_before_validation(self):
        contour = _raster_contour(
            [(306, 263), (134, 216), (291, 188), (301, 219)]
        )
        cleaned, indices = geometry.simplify_closed_contour_indices(
            contour, 3.6
        )
        rough_edge_lengths = [
            geometry.distance(
                cleaned[indices[index]],
                cleaned[indices[(index + 1) % len(indices)]],
            )
            for index in range(len(indices))
        ]

        self.assertEqual(4, len(indices))
        self.assertLess(min(rough_edge_lengths), 32.0)
        self.assertEqual(
            indices,
            geometry._prune_short_corner_pairs(
                cleaned, indices, minimum_edge_length=32.0
            ),
        )

    def test_clean_triangle_remains_triangle(self):
        contour = _raster_contour(
            [(80, 40), (300, 110), (140, 320)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(3, len(vertices))

    def test_merges_invalid_short_edge_instead_of_rejecting_triangle(self):
        # Reproduces the narrow-piece failure seen on camera: RDP retains a
        # fourth point at the clipped tip, but its fitted 25 px edge is below
        # the detector's 32 px physical minimum.
        contour = _raster_contour(
            [(100, 10), (300, 120), (20, 100), (25, 75)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(3, len(vertices))

    def test_merges_invalid_short_edge_instead_of_rejecting_quad(self):
        # Reproduces the long top piece: threshold stair-stepping creates a
        # fifth 23 px side even though the physical piece is a quadrilateral.
        contour = _raster_contour(
            [(268, 22), (473, 41), (470, 107), (188, 53), (190, 30)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(4, len(vertices))

    def test_merges_split_triangle_across_recovery_path(self):
        # Live camera example: lower-epsilon recovery finds four long
        # segments, but the extra point has a 177.9 degree interior angle.
        # It is a split side of the large physical triangle, not a corner.
        contour = _raster_contour(
            [(314, 169), (455, 310), (272, 332), (115, 345)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(3, len(vertices))

    def test_merges_current_live_triangle_split_at_176_degrees(self):
        # Exact vertices from the current OpenMV frame. The detector reported
        # four sides because the final 4->3 near-straight cleanup was missing.
        contour = _raster_contour(
            [(281, 168), (437, 289), (260, 336), (116, 366)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=28.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(3, len(vertices))

    def test_valid_wide_quadrilateral_below_angle_limit_is_preserved(self):
        # The final cleanup is restricted to near-straight (170+ degree)
        # points, so a genuine wide quadrilateral is still four-sided.
        vertices = [(40, 40), (250, 70), (230, 150), (80, 180)]

        reduced = geometry._merge_collinear_fitted_vertices(
            vertices,
            minimum_edge_length=28.0,
            minimum_area=600.0,
        )

        self.assertEqual(4, len(reduced))

    def test_merges_nearly_collinear_fifth_point_to_quad(self):
        contour = _raster_contour(
            [(219, 57), (401, 59), (436, 65), (435, 134), (137, 97)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(4, len(vertices))

    def test_recovery_merges_fifth_point_but_keeps_true_short_side(self):
        # Live-camera regression: the primary fit merges the 28 px tip and
        # becomes a triangle.  Lower-epsilon recovery restores that true short
        # side, but also exposes a nearly-collinear fifth point on a long side.
        contour = _raster_contour(
            [(279, 125), (187, 197), (87, 285), (67, 265), (125, 163)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=28.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(4, len(vertices))
        edge_lengths = [
            geometry.distance(
                vertices[index],
                vertices[(index + 1) % len(vertices)],
            )
            for index in range(len(vertices))
        ]
        self.assertGreaterEqual(min(edge_lengths), 28.0)

    def test_regular_pentagon_remains_pentagon(self):
        contour = _raster_contour(
            [(160, 30), (280, 115), (235, 260), (85, 260), (40, 115)]
        )

        vertices, reason = geometry.polygon_from_contour(
            contour,
            rdp_epsilon=6.0,
            minimum_edge_length=32.0,
            minimum_area=600.0,
        )

        self.assertIsNone(reason)
        self.assertEqual(5, len(vertices))


if __name__ == "__main__":
    unittest.main()
