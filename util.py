# This Python file uses the following encoding: utf-8

import logging
import sys
import cv2
import numpy


def show_image(img, title="Imagen"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img, path, suffix=""):
    cv2.imwrite(path + suffix + "_modified.jpg", img)


def crop_image(image, start, dimensions):
    cropped_image = image[start[1]:start[1] + dimensions[1], start[0]:start[0] + dimensions[0]].copy()
    return cropped_image


def configure_logger():
    global logger
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def closest_value(input_list, value):
    arr = numpy.asarray(input_list)
    i = (numpy.abs(arr - value)).argmin()
    return arr[i]