import tango
from enum import Enum


class NetworkStatus(Enum):
    ELLIPSE = 'ellipse'
    CIRCLE = 'circle'
    BAD_BEAM = 'bad_beam'


class Detector:
    """
    MeasStatus - tensorflow detection class, enum;
    X, Y - coordinates of mass center (int);
    Square - square of fig (float);
    Eccentricity (float)
    Distance - distance between (X,Y) and center of net;
    Timestamp;
    """
    def __init__(self, status, timestamp, ec, square, x, y, dist, image):
        self.__status = status
        self.__square = square
        self.__ec = ec
        self.__dist = dist
        self.__timestamp = timestamp
        self.__x = x
        self.__y = y
        self.__image = image

    @classmethod
    def verify_numeric(cls, a):
        return type(a) in (int, float)

    @staticmethod
    def set_device(source):
        return tango.DeviceProxy(source)

    @property
    def all_attr(self):
        return [("MeasStatus", self.__status), ("Timestamp", self.timestamp),
                ("Eccentricity", self.__ec), ("Square", self.__square),
                ("X", self.__x), ("Y", self.__y), ("Distance", self.__dist),
                ("Image", self.__image)]

    @all_attr.setter
    def all_attr(self, *args):
        bool_list = []
        for value in args:
            bool_list.append(self.verify_numeric(value))

        if set(bool_list) == {True}:
            self.__status = args[0]
            self.__timestamp = args[1]
            self.__ec = args[2]
            self.__square = args[3]
            self.__x = args[4]
            self.__y = args[5]
            self.__dist = args[6]
            self.__image = args[7]
        else:
            raise TypeError('Значения аттрибутов должны быть (int, float)')

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, status):
        self.__status = status

    @property
    def square(self):
        return self.__square

    @square.setter
    def square(self, square):
        self.__square = square

    @property
    def timestamp(self):
        return self.__timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        self.__timestamp = timestamp

    @property
    def ec(self):
        return self.__ec

    @ec.setter
    def ec(self, ec):
        self.__ec = ec

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def dist(self):
        return self.__dist

    @dist.setter
    def dist(self, dist):
        self.__dist = dist

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        self.__image = image

# if __name__ == "__main__":
#     device = Detector.set_device("training/collectords/bn.lumin.meas")
#     train_device = Detector(0, 0, 0, 0, 0, 0, 0)
#     list_of_attr = [("MeasStatus", Detector.status), ("Timestamp", Detector.timestamp),
#                     ("Eccentricity", Detector.ec), ("Square", Detector.square),
#                     ("X", Detector.x), ("Y", Detector.y), ("Distance", Detector.dist)]
#
#     device.write_attributes(list_of_attr)

    #
    #
