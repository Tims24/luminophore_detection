from tango.server import Device, attribute, command, device_property
from tango import AttrQuality, AttrWriteType, DispLevel
import datetime


class LuminophoreDetector(Device):
    def __init__(self, status, timestamp, ec, square, x, y, dist, image):
        self.__status = status
        self.__ec = ec
        self.__square = square
        self.__x = x
        self.__y = y
        self.__dist = dist
        self.__image = image

    host = device_property(dtype=str)

    @attribute()
    def status(self):
        return self.__status

    @attribute()
    def timestamp(self):
        return datetime.datetime.now().timestamp()

    @attribute()
    def ec(self):
        return self.__ec

    @attribute()
    def square(self):
        return self.__square

    @attribute()
    def x(self):
        return self.__x

    @attribute()
    def y(self):
        return self.__y

    @attribute()
    def dist(self):
        return self.__dist

    image = attribute(label='Image', dtype=((int,),),
                      max_dim_x=1024, max_dim_y=1024,
                      fget="get_image")

    def get_image(self):
        return self.__image
