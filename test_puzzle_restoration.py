"""Deterministic host-side tests for the camera-driven puzzle simulator."""

import json
import math
import unittest

import puzzle_restoration as restoration


def _place(vertices, center, angle):
    centroid = restoration.polygon_centroid(vertices)
    return restoration.transform_polygon(
        vertices,
        restoration.compose_transform(
            restoration.rigid_transform(angle, center[0], center[1]),
            restoration.rigid_transform(0.0, -centroid[0], -centroid[1]),
        ),
    )


class PuzzleRestorationTests(unittest.TestCase):
    def setUp(self):
        # A deterministic 200x100 rectangle split by one slanted internal edge.
        left_source = [(0, 0), (70, 0), (100, 100), (0, 100)]
        right_source = [(70, 0), (200, 0), (200, 100), (100, 100)]
        self.left = _place(left_source, (155, 105), math.radians(31))
        self.right = _place(right_source, (455, 115), math.radians(-47))
        self.payload = {
            "status": "ok",
            "count": 2,
            "coordinate_system": {
                "origin_px": [0, 0],
                "x_direction": "right",
                "y_direction": "down",
                "vertex_order": "clockwise",
            },
            "polygons": [
                {"id": 1, "side_count": 4, "vertices_px": self.left},
                {"id": 2, "side_count": 4, "vertices_px": self.right},
            ],
        }

    def test_parses_openmv_json_line(self):
        line = "DEBUG " + json.dumps(self.payload)
        parsed = restoration.payload_from_json_line(line)
        pieces = restoration.pieces_from_payload(parsed)

        self.assertEqual([1, 2], [piece.piece_id for piece in pieces])
        self.assertEqual(4, len(pieces[0].vertices))

    def test_pasted_terminal_output_uses_latest_complete_json(self):
        older = dict(self.payload)
        older["threshold"] = 170
        latest = dict(self.payload)
        latest["threshold"] = 181
        pasted = (
            "POLYGON_DETECTOR_READY\n"
            + json.dumps(older, ensure_ascii=False)
            + "\nFPS=7.4\n"
            + json.dumps(latest, ensure_ascii=False, indent=2)
            + "\n"
        )

        parsed = restoration.payload_from_text(pasted)

        self.assertEqual(181, parsed["threshold"])
        self.assertEqual(2, len(restoration.pieces_from_payload(parsed)))

    def test_restores_rectangle_into_lower_half(self):
        pieces = restoration.pieces_from_payload(self.payload)
        solution = restoration.solve_puzzle(
            pieces, frame_size=(640, 480), divider_y=240
        )
        restored = [
            restoration.transform_polygon(piece.vertices, transform)
            for piece, transform in zip(
                solution.pieces, solution.transforms
            )
        ]
        all_points = [point for polygon in restored for point in polygon]
        width = max(point[0] for point in all_points) - min(
            point[0] for point in all_points
        )
        height = max(point[1] for point in all_points) - min(
            point[1] for point in all_points
        )

        self.assertAlmostEqual(200.0, width, places=5)
        self.assertAlmostEqual(100.0, height, places=5)
        self.assertGreaterEqual(min(point[1] for point in all_points), 240.0)
        self.assertAlmostEqual(
            0.0,
            restoration.polygon_overlap_area(restored[0], restored[1]),
            places=5,
        )
        self.assertEqual(2, len(solution.motions))

    def test_animation_finishes_at_exact_solution(self):
        pieces = restoration.pieces_from_payload(self.payload)
        solution = restoration.solve_puzzle(pieces)
        for piece, transform in zip(solution.pieces, solution.transforms):
            interpolated = restoration.interpolate_polygon(
                piece.vertices, transform, 1.0
            )
            expected = restoration.transform_polygon(
                piece.vertices, transform
            )
            for actual_point, expected_point in zip(interpolated, expected):
                self.assertAlmostEqual(
                    expected_point[0], actual_point[0], places=7
                )
                self.assertAlmostEqual(
                    expected_point[1], actual_point[1], places=7
                )

    def test_restores_four_piece_radial_cut(self):
        top_left, top_right = (0, 0), (220, 0)
        bottom_right, bottom_left = (220, 120), (0, 120)
        top, right = (80, 0), (220, 75)
        bottom, left, center = (135, 120), (0, 45), (105, 58)
        source = [
            (top_left, top, center, left),
            (top, top_right, right, center),
            (center, right, bottom_right, bottom),
            (left, center, bottom, bottom_left),
        ]
        placements = [
            ((100, 70), 0.35),
            ((270, 75), -1.0),
            ((430, 75), 1.35),
            ((560, 75), -0.55),
        ]
        pieces = tuple(
            restoration.Piece(
                index + 1,
                tuple(_place(polygon, placement[0], placement[1])),
            )
            for index, (polygon, placement) in enumerate(
                zip(source, placements)
            )
        )

        solution = restoration.solve_puzzle(pieces)
        restored = [
            restoration.transform_polygon(piece.vertices, transform)
            for piece, transform in zip(
                solution.pieces, solution.transforms
            )
        ]
        overlap = sum(
            restoration.polygon_overlap_area(
                restored[first], restored[second]
            )
            for first in range(4)
            for second in range(first + 1, 4)
        )

        self.assertAlmostEqual(220.0, solution.target_bounds[2], places=5)
        self.assertAlmostEqual(120.0, solution.target_bounds[3], places=5)
        self.assertAlmostEqual(0.0, overlap, places=5)
        self.assertEqual(3, len(solution.matches))

    def test_question_one_recognises_fixed_shapes_and_pixel_scale(self):
        pixels_per_cm = 26.0
        template_order = [2, 0, 3, 1]
        poses = [
            ((90, 70), 0.50),
            ((260, 80), -1.10),
            ((430, 75), 1.40),
            ((560, 80), -0.70),
        ]
        noise = [
            ((0.4, -0.5), (-0.7, 0.2), (0.5, 0.6), (-0.2, -0.3)),
            ((-0.3, 0.6), (0.8, -0.4), (-0.5, 0.1), (0.2, -0.2)),
            ((0.6, 0.1), (-0.4, -0.7), (0.3, 0.4), (-0.6, 0.2)),
            ((-0.5, 0.3), (0.4, -0.6), (0.2, 0.5)),
        ]
        pieces = []
        for piece_id, template_index in enumerate(template_order, 1):
            template = restoration.QUESTION_ONE_TEMPLATES[template_index]
            scaled = [
                (
                    point[0] * pixels_per_cm,
                    point[1] * pixels_per_cm,
                )
                for point in template.vertices_cm
            ]
            placed = _place(
                scaled,
                poses[piece_id - 1][0],
                poses[piece_id - 1][1],
            )
            perturbed = [
                (
                    point[0] + noise[piece_id - 1][index][0],
                    point[1] + noise[piece_id - 1][index][1],
                )
                for index, point in enumerate(placed)
            ]
            shift = piece_id % len(perturbed)
            perturbed = perturbed[shift:] + perturbed[:shift]
            pieces.append(
                restoration.Piece(piece_id, tuple(perturbed))
            )

        solution = restoration.solve_question_one_fixed(pieces)
        identity_by_piece = {
            match.piece_id: match.template_id
            for match in solution.template_matches
        }

        self.assertEqual(
            {1: "F3", 2: "F1", 3: "F4", 4: "F2"},
            identity_by_piece,
        )
        self.assertAlmostEqual(
            pixels_per_cm, solution.pixels_per_cm, delta=0.2
        )
        self.assertEqual(tuple(pieces), solution.pieces)
        self.assertAlmostEqual(0.5, solution.clearance_cm)
        clearance_px = (
            solution.clearance_cm * solution.pixels_per_cm * 0.99
        )
        for first in range(4):
            for second in range(first + 1, 4):
                self.assertIsNone(
                    restoration._clearance_translation(
                        solution.target_polygons[first],
                        solution.target_polygons[second],
                        (1.0, 1.0),
                        clearance_px,
                    )
                )
        self.assertEqual("fixed_question_one", solution.mode)

    def test_question_one_accepts_real_low_precision_camera_payload(self):
        payload = {
            "status": "ok",
            "count": 4,
            "polygons": [
                {
                    "id": 1,
                    "side_count": 4,
                    "vertices_px": [
                        [357, 91],
                        [482, 133],
                        [531, 223],
                        [506, 239],
                    ],
                },
                {
                    "id": 2,
                    "side_count": 4,
                    "vertices_px": [
                        [298, 91],
                        [443, 232],
                        [472, 334],
                        [237, 145],
                    ],
                },
                {
                    "id": 3,
                    "side_count": 4,
                    "vertices_px": [
                        [102, 177],
                        [127, 218],
                        [120, 276],
                        [69, 275],
                    ],
                },
                {
                    "id": 4,
                    "side_count": 3,
                    "vertices_px": [
                        [188, 140],
                        [396, 348],
                        [142, 301],
                    ],
                },
            ],
        }

        solution = restoration.solve_question_one_fixed(
            restoration.pieces_from_payload(payload)
        )
        identity_by_piece = {
            match.piece_id: match.template_id
            for match in solution.template_matches
        }

        self.assertEqual(
            {1: "F3", 2: "F4", 3: "F1", 4: "F2"},
            identity_by_piece,
        )
        self.assertAlmostEqual(
            27.509, solution.pixels_per_cm, places=3
        )
        self.assertTrue(solution.warnings)
        self.assertEqual(
            ["低", "高", "高", "高"],
            [match.confidence for match in solution.template_matches],
        )
        self.assertGreater(
            solution.template_matches[0].observation_max_cm, 2.0
        )
        self.assertTrue(
            all(
                0.0 <= x <= 640.0 and 0.0 <= y <= 480.0
                for polygon in solution.target_polygons
                for x, y in polygon
            )
        )
        for piece, transform, target_polygon in zip(
            solution.pieces,
            solution.transforms,
            solution.target_polygons,
        ):
            restored = restoration.transform_polygon(
                piece.vertices, transform
            )
            for actual, expected in zip(restored, target_polygon):
                self.assertAlmostEqual(actual[0], expected[0], places=6)
                self.assertAlmostEqual(actual[1], expected[1], places=6)
            original_edges = [
                restoration.distance(first, second)
                for first, second in restoration.polygon_edges(
                    piece.vertices
                )
            ]
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                animated = restoration.interpolate_polygon(
                    piece.vertices, transform, fraction
                )
                animated_edges = [
                    restoration.distance(first, second)
                    for first, second in restoration.polygon_edges(animated)
                ]
                for original, current in zip(
                    original_edges, animated_edges
                ):
                    self.assertAlmostEqual(original, current, places=6)


if __name__ == "__main__":
    unittest.main()
