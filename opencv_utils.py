import cv2
import numpy as np
import math


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):

                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)

        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]),
                                          None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def cal_pt_distance(pt1, pt2):
    dist = math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))
    return dist


def get_contours(img, img_contour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(img_contour, cnt, -1, (0, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            # print(f'Площадь = {area}, длина Эллипса = {peri}')
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h, = cv2.boundingRect(approx)

            # cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # ellipse parameters a - big semiaxis, b - small semiaxis, c - focal length
            # a = len(cv2.line(img_contour, (x,y), (x+w, y+h), (0, 0, 0), 1)) / 2

            cv2.line(img_contour, (x, y), (x + w, y + h), (0, 0, 0), 1)

            cv2.putText(img_contour, f"S = {round(area, 2)}, l = {round(peri, 2)}",
                        (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 0, 0), 1)
            return area, peri, x, y, w, h, cnt


def empty(a):
    pass


def invert(img, name):
    img_inv = (255-img)
    cv2.imwrite(name, img_inv)
    return img_inv


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


def one_shot(source):

    r = 0
    camera = cv2.VideoCapture(source)
    while camera.isOpened():
        ret, frame = camera.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) % 256 == 32:
            r = cv2.selectROI("select the area", frame)
            cv2.destroyAllWindows()
            break
    return r


def draw_contours(img, img_contour):
    try:

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                (x1, y1), (ma, MA), angle = cv2.fitEllipse(cnt)
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(img_contour, ellipse, (0, 0, 255), 2)
                return (x1, y1), (ma, MA), angle
    except TypeError:
        print("Сломалось(")