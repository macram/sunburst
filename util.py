# This Python file uses the following encoding: utf-8

import logging
import sys
import cv2


def show_image(img, title="Imagen"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img, path, suffix=""):
    cv2.imwrite(path + suffix + "_modified.jpg", img)


def configure_logger():
    global logger
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
