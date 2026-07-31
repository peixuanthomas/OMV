"""Temporary raw-FOV viewer for the fixed OpenMV camera.

This intentionally does not load or apply calibration. It shows the exact
VGA grayscale coverage used by the production detector so the camera focus,
mount, and A4 margin can be checked before calibration.
"""

import time

from openmv_edge_detection import init_camera


GRID_COLUMNS = 8
GRID_ROWS = 6
SAFE_MARGIN_PX = 12
REPORT_EVERY_FRAMES = 30


def _draw_shadowed_text(frame, x, y, text):
    frame.draw_string((x + 1, y + 1), text, color=0, scale=1)
    frame.draw_string((x, y), text, color=255, scale=1)


def draw_fov_overlay(frame):
    width = frame.width()
    height = frame.height()

    for column in range(1, GRID_COLUMNS):
        x = int(round(column * (width - 1) / GRID_COLUMNS))
        frame.draw_line((x, 0, x, height - 1), color=190, thickness=1)
    for row in range(1, GRID_ROWS):
        y = int(round(row * (height - 1) / GRID_ROWS))
        frame.draw_line((0, y, width - 1, y), color=190, thickness=1)

    frame.draw_rectangle(
        (
            SAFE_MARGIN_PX,
            SAFE_MARGIN_PX,
            width - (2 * SAFE_MARGIN_PX),
            height - (2 * SAFE_MARGIN_PX),
        ),
        color=255,
        thickness=2,
    )
    center_x = width // 2
    center_y = height // 2
    frame.draw_line(
        (center_x - 16, center_y, center_x + 16, center_y),
        color=255,
        thickness=2,
    )
    frame.draw_line(
        (center_x, center_y - 16, center_x, center_y + 16),
        color=255,
        thickness=2,
    )
    _draw_shadowed_text(
        frame,
        4,
        4,
        "RAW FOV %dx%d" % (width, height),
    )
    _draw_shadowed_text(
        frame,
        4,
        height - 14,
        "KEEP A4 >= %dPX FROM BORDER" % SAFE_MARGIN_PX,
    )


def main():
    camera = init_camera()
    clock = time.clock()
    frame_count = 0

    print("FOV_PREVIEW_READY")
    print("MODE=RAW_GRAYSCALE RESOLUTION=%dx%d" % (
        camera.width(),
        camera.height(),
    ))
    print("CAMERA_HEIGHT_NOMINAL_MM=295 RANGE_MM=290..300")

    try:
        while True:
            clock.tick()
            frame = camera.snapshot()
            draw_fov_overlay(frame)
            camera.flush()

            frame_count += 1
            if frame_count >= REPORT_EVERY_FRAMES:
                print(
                    "FOV FPS=%.2f gain=%.2fdB exposure=%dus"
                    % (
                        clock.fps(),
                        camera.gain_db(),
                        camera.exposure_us(),
                    )
                )
                frame_count = 0
    except KeyboardInterrupt:
        camera.flush()
        print("FOV_PREVIEW_STOPPED")


if __name__ == "__main__":
    main()
