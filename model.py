# This Python file uses the following encoding: utf-8
from array import ArrayType

import cv2

class Image(object):
    img = None
    red_ink_img = None
    groups_array = None
    center_x = None
    center_y = None
    circle_radius = None
    measured_bursts = []

    def __init__(self, path, img):
        self.path = path
        self.img = img

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

    def get_description(self):
        string = self.i.__str__() + " -> "
        string += self.theta.__str__()
        string += " -> "
        string += self.white_pixels.__str__()
        string += " -> Height: "
        string += self.height.__str__()
        string += " -> Base: "
        string += self.base.__str__()
        return string
