# This Python file uses the following encoding: utf-8
from array import ArrayType

import cv2

class Image(object):
    image_index = None
    img = None
    path = None
    red_ink_img = None
    groups_array = None
    center_x = None
    center_y = None
    circle_radius = None
    measured_bursts = []

    def __init__(self, path, img):
        self.path = path
        self.img = img

    def get_description(self, human = True, headers = False):
        if human is True:
            return self.get_human_description()
        else:
            return self.get_csv_description(headers)

    def get_human_description(self):
        string = "Image n. " + self.image_index.__str__() + "\n"
        string += "===============\n"
        string += "Path: \"" + self.path + "\"\n"
        string += "Measured " + len(self.measured_bursts).__str__() + " bursts\n"
        sorted_bursts = sorted(self.measured_bursts, key=MeasuredBurst.theta)
        for burst in sorted_bursts:
            string += burst.get_description()
        return string

    def get_csv_description(self, headers=True):
        if headers is True:
            string = "Index" + ","
            string += "Angle" + ","
            string += "Area" + ","
            string += "Height" + ","
            string += "Base"+ "\n"
        sorted_bursts = sorted(self.measured_bursts, key=MeasuredBurst.theta)
        for burst in sorted_bursts:
            string += burst.get_csv_description()
        return string

class MeasuredBurst(object):
    i = None
    r = None
    theta = None
    white_pixels = None
    height = None
    base = None

    def __init__(self, i, r, theta, white_pixels, height, base):
        self.i = i
        self.r = r
        self.theta = theta
        self.white_pixels = white_pixels
        self.height = height
        self.base = base

    def get_description(self, human = True):
        if human is True:
            return self.get_human_description()
        else:
            return self.get_csv_description()

    def get_human_description(self):
        string = self.i.__str__() + " -> "
        string += self.theta.__str__()
        string += " -> Area: "
        string += self.white_pixels.__str__()
        string += " -> Height: "
        string += self.height.__str__()
        string += " -> Base: "
        string += self.base.__str__()
        string += "\n"
        return string

    def get_csv_description(self):
        string = self.i.__str__() + ","
        string += self.theta.__str__() + ","
        string += self.white_pixels.__str__() + ","
        string += self.height.__str__() + ","
        string += self.base.__str__() + "\n"
        return string

    def theta(self):
        return self.theta
