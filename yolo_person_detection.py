import csi
import ml
import time
from ml.postprocessing.darknet import YoloLC


MODEL_PATH = "/rom/yolo_lc_192.tflite"
CONFIDENCE_THRESHOLD = 0.40
BOX_COLOR = (0, 255, 0)


def load_model():
    try:
        model = ml.Model(
            MODEL_PATH,
            postprocess=YoloLC(threshold=CONFIDENCE_THRESHOLD),
        )
        print("YOLO model loaded:", MODEL_PATH)
        print(model)
        return model
    except Exception as error:
        print("MODEL_LOAD_ERROR:", error)
        raise


def main():
    camera = csi.CSI()
    camera.reset()
    camera.pixformat(csi.RGB565)
    camera.framesize(csi.VGA)
    camera.snapshot(time=2000)

    model = load_model()
    clock = time.clock()

    while True:
        clock.tick()
        image = camera.snapshot()

        try:
            detections_by_class = model.predict([image])
        except Exception as error:
            print("INFERENCE_ERROR:", error)
            raise

        detection_count = 0
        for class_index, class_detections in enumerate(detections_by_class):
            if not class_detections:
                continue

            rectangles = [rectangle for rectangle, score in class_detections]
            scores = [score for rectangle, score in class_detections]
            labels = [model.labels[class_index] for _ in class_detections]
            colors = [BOX_COLOR for _ in class_detections]

            ml.utils.draw_predictions(
                image,
                rectangles,
                labels,
                colors,
                scores=scores,
                format=None,
            )
            detection_count += len(class_detections)

        print("FPS: %.2f, persons: %d" % (clock.fps(), detection_count))


main()
