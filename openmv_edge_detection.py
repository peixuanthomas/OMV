"""High-frame-rate object edge detection for OpenMV cameras.

This is the OpenMV ``csi`` port of ``04.Detecting/04.edges.py``. The camera
captures grayscale frames and ``find_edges()`` updates each frame in place, so
the processed image is sent to the OpenMV IDE preview without an extra image
allocation or display copy.
"""

import csi
import image
import time

import calibration_geometry as calibration
import a4_runtime


# Match the VGA resolution used by the original 04.Detecting example. The H7
# Plus has enough external memory for grayscale VGA with double buffering.
FRAME_SIZE = csi.VGA
FRAME_BUFFER_COUNT = 2
CAMERA_WARMUP_MS = 1500

# Keep the proven thresholds from 04.Detecting/04.edges.py.
EDGE_ALGORITHM = image.EDGE_CANNY
EDGE_THRESHOLD = (50, 80)

# Printing every frame can noticeably reduce both processing and IDE-preview
# throughput. A periodic report is enough to tune the camera.
FPS_REPORT_EVERY = 30


def init_camera():
    """Configure the OpenMV camera for low-copy grayscale processing."""
    camera = csi.CSI()
    camera.reset()
    camera.pixformat(csi.GRAYSCALE)
    camera.framesize(FRAME_SIZE)
    # Calibration is tied to this exact physical orientation. Make it
    # explicit instead of relying on sensor reset defaults.
    camera.hmirror(False)
    camera.vflip(False)
    camera.transpose(False)

    # Double buffering allows the sensor to capture the next frame while the
    # current frame is being processed. This call must follow framesize(),
    # because changing the format or frame size reallocates frame buffers.
    camera.framebuffers(FRAME_BUFFER_COUNT)

    # Let automatic exposure/gain settle before measuring FPS.
    camera.snapshot(time=CAMERA_WARMUP_MS)
    return camera


def load_runtime_calibration(camera, path=calibration.CONFIG_PATH):
    """Load calibration only when it matches this live camera geometry."""
    signature = calibration.camera_signature(camera)
    return calibration.load_calibration(path, signature=signature)


def rectify_image(frame, calibration_config):
    """Apply saved lens correction and the current frame's live A4 pose."""
    return a4_runtime.rectify_to_a4(frame, calibration_config)


def process_image(frame):
    """Replace a grayscale camera frame with its detected edges."""
    frame.find_edges(EDGE_ALGORITHM, threshold=EDGE_THRESHOLD)
    return frame


def main():
    camera = init_camera()
    calibration_config, calibration_error = load_runtime_calibration(camera)
    clock = time.clock()
    frame_count = 0

    print(
        "Edge detection ready: %dx%d, Canny thresholds=%s"
        % (camera.width(), camera.height(), str(EDGE_THRESHOLD))
    )
    if calibration_config is None:
        print("CALIBRATION_REQUIRED: %s" % calibration_error)
    else:
        print("LENS_CALIBRATION_OK A4_POSE=DETECTED_EACH_FRAME")

    try:
        while True:
            clock.tick()
            frame = camera.snapshot()
            if calibration_config is not None:
                try:
                    rectify_image(frame, calibration_config)
                except Exception as error:
                    print("A4_POSE_REQUIRED: %s" % str(error))
                    camera.flush()
                    continue
            process_image(frame)

            frame_count += 1
            if frame_count >= FPS_REPORT_EVERY:
                print("Edge FPS: %.2f" % clock.fps())
                frame_count = 0
    except KeyboardInterrupt:
        # Preserve the last processed frame in the IDE after stopping.
        camera.flush()
        print("Edge detection stopped")


if __name__ == "__main__":
    main()
