import cv2
import numpy as np
from opencv_utils import stack_images, cal_pt_distance, empty, one_shot, draw_contours
import math

if __name__ == "__main__":
    kernel = np.ones((5, 5))

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
        imgContour = img.copy()
        imgEmpty = np.zeros_like(img)
        imgEmpty_copy = imgEmpty.copy()

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

        # ellipse draw
        try:
            (x1, y1), (ma, MA), angle = draw_contours(imgResult, imgContour)

            a = MA / 2
            b = ma / 2
            perimeter = 4 * (math.pi * a * b + (a - b) ** 2) / (a + b)

            cv2.circle(imgContour, (int(x1), int(y1)), 5, (255, 0, 0), -1)

            eccentricity = math.sqrt(pow(a, 2) - pow(b, 2))
            eccentricity = round(eccentricity / a, 2)

            cv2.circle(imgContour, (x+w//2, y+h//2), 5, (0, 0, 255), 2)
            # cv2.rectangle(temp, (x11, y11), (x11+w11, y11+h11), (0, 255, 0), 2)
            cv2.line(imgContour, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.line(imgContour, (x, y+h), (x+w, y), (255, 0, 0), 1)
            cv2.line(imgContour, (x+w//2, y+h//2), (int(x1), int(y1)), (0, 255, 255), 2)

            x = cv2.getTrackbarPos("x", "TrackBars")
            y = cv2.getTrackbarPos("y", "TrackBars")

            cv2.putText(imgEmpty, f"a={round(a, 1)}, b={round(b, 1)}, c={round(math.sqrt(a ** 2 - b ** 2), 1)}",
                        (x - 70, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(imgEmpty, f"S={round(math.pi * a * b, 2)}, L={round(perimeter, 1)}",
                        (x - 70, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(imgEmpty, f"e={round(math.sqrt(a ** 2 - b ** 2) / a, 3)}",
                        (x - 70, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(imgEmpty, f"dist={round(cal_pt_distance((x+w//2, y+h//2), (int(x1), int(y1))),2)}",
                        (x-70, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

            # image show
            stacked = stack_images(0.5, ([imgContour, imgEmpty],
                                         [imgResult, img]))
            cv2.imshow("Stack", stacked)

            if cv2.waitKey(1) % 256 == 27:
                cv2.destroyAllWindows()
                break

        except TypeError:
            pass


