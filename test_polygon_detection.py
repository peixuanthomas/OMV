"""Host-side regression tests for placement-dependent detector helpers."""

import sys
import types
import unittest


# ``polygon_detection`` imports the OpenMV-only camera module at import time.
# Geometry and mask helpers under test do not use it, so a small host stub is
# enough to load the module without weakening the on-device import path.
openmv_edges = types.ModuleType("openmv_edge_detection")
openmv_edges.EDGE_THRESHOLD = (50, 80)
openmv_edges.init_camera = None
openmv_edges.load_runtime_calibration = None
openmv_edges.process_image = None
openmv_edges.rectify_image = None
sys.modules.setdefault("openmv_edge_detection", openmv_edges)

import polygon_detection as detection


class _Mask:
    def __init__(self, width, height, foreground):
        self._width = width
        self._height = height
        self._foreground = set(foreground)

    def width(self):
        return self._width

    def height(self):
        return self._height

    def get_pixel(self, point):
        x, y = point
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return None
        return 255 if (x, y) in self._foreground else 0


class _Blob:
    rect = (10, 40, 21, 21)
    cx = 20
    cy = 50


class PolygonDetectionTests(unittest.TestCase):
    def test_seed_stays_with_blob_when_bounding_rectangles_overlap(self):
        foreground = set()

        # Target triangle. Its bounding rectangle starts at y=40, but the
        # first target pixel on that row is at the far right.
        for y in range(40, 61):
            left = 30 - (y - 40)
            for x in range(left, 31):
                foreground.add((x, y))

        # A separate component crosses the target's bounding rectangle near
        # its top-left. The old top-to-bottom scan selected its interior and
        # returned boundary_trace_failed with a zero-length contour.
        for y in range(35, 44):
            for x in range(0, 26):
                foreground.add((x, y))

        mask = _Mask(64, 64, foreground)
        blob = _Blob()
        seed = detection._find_blob_boundary_seed(mask, blob, blob.rect)

        self.assertEqual((20, 50), seed)
        self.assertEqual(0, mask.get_pixel((seed[0] - 1, seed[1])))
        contour, closed = detection.trace_outer_boundary(mask, seed, 500)
        self.assertTrue(closed)
        self.assertGreater(len(contour), 3)


if __name__ == "__main__":
    unittest.main()
