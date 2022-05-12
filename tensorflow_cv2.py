import cv2
import numpy as np
from utils import stack_images, cal_pt_distance, empty, one_shot, draw_contours, visualize_boxes_and_labels_on_image_array
import math
import joblib
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
import os
from tango_utils.device import Detector
import datetime

paths = joblib.load("data/paths.joblib")
files = joblib.load("data/files.joblib")

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


if __name__ == "__main__":
    kernel = np.ones((5, 5))

    device = Detector.set_device("training/collectords/bn.lumin.meas")
    attr_device = Detector(0, 0, 0, 0, 0, 0, 0, 0)

    # camera parameters
    SOURCE = 0
    camera = cv2.VideoCapture(SOURCE)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # target draw
    r = one_shot(SOURCE)

    """
    Track bar to change values of parameters.
    Ex: min val, max val, position of target 
    """
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 540, 340)
    cv2.createTrackbar("Val min", "TrackBars", 100, 255, empty)
    cv2.createTrackbar("Val max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar('x', "TrackBars", r[0], width, empty)
    cv2.createTrackbar('y', "TrackBars", r[1], height, empty)

    # coordinates of target
    x, y, w, h = r[0], r[1], r[2], r[3]

    while camera.isOpened():

        ret, img = camera.read()

        # tensorflow part
        image_np = np.array(img)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        label, _ = visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

        imgEmpty = np.zeros_like(img)

        # image preprocess
        imgBlur = cv2.GaussianBlur(img, (9, 9), 4)
        imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

        v_min = cv2.getTrackbarPos("Val min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val max", "TrackBars")

        lower = np.array([0, 0, v_min])
        upper = np.array([179, 255, v_max])

        mask = cv2.inRange(imgHSV, lower, upper)
        imgCanny = cv2.Canny(mask, 70, 70)
        imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
        imgResult = cv2.erode(imgDial, kernel, iterations=3)
        # imgResult = imgResult[:, :, 2]

        # ellipse draw
        try:
            (x1, y1), (ma, MA), angle = draw_contours(imgResult, image_np_with_detections)

            a = MA / 2
            b = ma / 2
            perimeter = 4 * (math.pi * a * b + (a - b) ** 2) / (a + b)

            cv2.circle(image_np_with_detections, (int(x1), int(y1)), 5, (255, 0, 0), -1)

            eccentricity = math.sqrt(pow(a, 2) - pow(b, 2))
            eccentricity = round(eccentricity / a, 2)
            square = round(math.pi * a * b, 2)
            dist = round(cal_pt_distance((x+w//2, y+h//2), (int(x1), int(y1))), 2)

            cv2.circle(image_np_with_detections, (x+w//2, y+h//2), 5, (0, 0, 255), 2)
            # cv2.rectangle(temp, (x11, y11), (x11+w11, y11+h11), (0, 255, 0), 2)
            cv2.line(image_np_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.line(image_np_with_detections, (x, y+h), (x+w, y), (255, 0, 0), 1)
            cv2.line(image_np_with_detections, (x+w//2, y+h//2), (int(x1), int(y1)), (0, 255, 255), 2)

            x = cv2.getTrackbarPos("x", "TrackBars")
            y = cv2.getTrackbarPos("y", "TrackBars")

            cv2.putText(imgEmpty, f"a={round(a, 1)}, b={round(b, 1)}, c={round(math.sqrt(a ** 2 - b ** 2), 1)}",
                        (x - 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(imgEmpty, f"S={square}, L={round(perimeter, 1)}",
                        (x - 70, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(imgEmpty, f"e={round(eccentricity, 3)}",
                        (x - 70, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(imgEmpty, f"dist={dist}",
                        (x-70, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

            # image show
            stacked = stack_images(0.5, ([imgResult, img],
                                         [image_np_with_detections, imgEmpty]))
            cv2.imshow("Stack", stacked)

            grayImg = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2GRAY)
            if label in ("ellipse", "circle"):
                attr_device.status = 1
                attr_device.timestamp = datetime.datetime.now().timestamp()
                attr_device.ec = eccentricity
                attr_device.square = square
                attr_device.x = x1
                attr_device.y = y1
                attr_device.dist = dist
                attr_device.image = grayImg
            else:
                attr_device.status = 0
                attr_device.timestamp = datetime.datetime.now().timestamp()
                attr_device.ec = eccentricity
                attr_device.square = square
                attr_device.x = x1
                attr_device.y = y1
                attr_device.dist = dist
                attr_device.image = grayImg

            if cv2.waitKey(2000) % 256 == 27:
                cv2.destroyAllWindows()
                cv2.imwrite('result.jpg', grayImg)
                device.write_attributes([("MeasStatus", attr_device.status), ("Timestamp", attr_device.timestamp),
                                         ("Eccentricity", attr_device.ec), ("Square", attr_device.square),
                                         ("X", attr_device.x), ("Y", attr_device.y),
                                         ("Distance", attr_device.dist), ('Image', grayImg)])
                break

        except TypeError:
            pass

        """
        MeasStatus, timestamp, ec, square, x, y, dist
        """
