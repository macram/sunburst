# This Python file uses the following encoding: utf-8

import argparse
import logging
import math
import os

import cv2
import numpy as np

import util

#### Parameters
# Error margin around the detected circle
errorMargin = 10


def grayscale_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def process_groups_array(image, groups_dictionary, center_x, center_y):
    i = 0
    for e in groups_dictionary:        # For each image in the dictionary... (we iterate on the keys)
        element = groups_dictionary[e] # ... we get the value, which is an array of tuples associated with each group, ...
        for tuple in element:          # ... and for each rectangle we do our deed.
            closest_rect = tuple[0]    # We use the first value in the tuple: the "closest" rectangle.
            bounding_rect = tuple[1]   # The second value would be the bounding ("straight") one.
            rect_center = closest_rect[0]
            r, theta = get_polar_from_cartesian(rect_center[0], rect_center[1], center_x, center_y)
            string = i.__str__() + " -> "
            string += theta.__str__()
            string += " -> "
            string += count_white_pixels(image, bounding_rect).__str__()
            util.logger.log(logging.INFO, string)
            i += 1


def count_white_pixels(image, rect):
    start = (rect[0], rect[1])       # (x, y)
    dimensions = (rect[2], rect[3])  # (Width, height)
    x = 0
    y = 0
    white_pixels = 0
    while x < dimensions[0]:
        while y < dimensions[1]:
            xx = start[0] + x
            yy = start[1] + y
            white_pixels += image[xx][yy]
            y += 1
        x += 1
    return white_pixels


def get_polar_from_cartesian(point_x, point_y, center_x, center_y):
    x = point_x - center_x  # Less than 0 <- Left hemisphere
    y = point_y - center_y  # Less than que 0 <- Upper hemisphere

    r = np.sqrt(x ** 2 + y ** 2)
    theta = (np.arctan2(x, y))  # Radians
    theta_deg = (math.degrees(theta) + 270) % 360

    return r, theta_deg


def circles(img, path=""):
    if isinstance(img, (np.ndarray, np.generic)):

        output = img.copy()
        gray = grayscale_image(img)

        util.logger.log(logging.DEBUG, "Detecting circles")
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.505, 100, param1=400, param2=150)

        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            intcircles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in intcircles:
                util.logger.log(logging.DEBUG, "Detected circle! Center: ({c_x},{c_y}), radius: {c_r}".format(c_x=x, c_y=y, c_r=r))
                mark_detected_circle(output, r, x, y)
                # cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

            cropped_img, center_x, center_y, r = crop_image(output, x, y, r)
            int_center_x = int(center_x)
            int_center_y = int(center_y)
            util.save_image(cropped_img, path, "_detectedcircle")

            if intcircles.size == 3:
                # First operations
                # Margin circles, used to discard measurements that are not immediately around the circle.
                margin_circle = get_error_margin_circle(cropped_img, int_center_x, int_center_y, r)
                # Red ink: color-wise masking
                red_ink = get_red_ink(cropped_img)
                # And now we just mask the exterior_circle image and the red_ink one, to know if that image has
                #     measurements.
                masked_mask = cv2.bitwise_and(margin_circle, red_ink)
                # showimage(cropped_img, path)

                # Saving the intermediate images to be inspected later.
                util.save_image(margin_circle, path, "_margincircle")
                util.save_image(red_ink, path, "_redink")
                util.save_image(masked_mask, path, "_maskedmask")

                # Doing the actual annotation group detection
                with_groups, groups_array = identify_groups(path, red_ink)
                process_groups_array(red_ink, groups_array, center_x, center_y)

                # Saving the image with the detected groups
                util.save_image(with_groups, path, "_withgroups")
                if np.count_nonzero(masked_mask) > 0:
                    return 1
                else:
                    return 0
            else:
                return -1
        else:
            return -2
    else:
        return -3


def mark_detected_circle(output, r, x, y):
    cv2.circle(output, (x, y), r, (0, 255, 255), 1)


def crop_image(img, center_x, center_y, r):
    crop_img = img[center_y - r - 30:center_y + r + 30, center_x - r - 30:center_x + r + 30]
    height, width, channels = crop_img.shape

    new_center_y = height / 2
    new_center_x = width / 2
    util.logger.log(logging.DEBUG,
                "Center of cropped circle: ({new_center_x},{new_center_y})".format(new_center_x=new_center_x,
                                                                                   new_center_y=new_center_y))
    return crop_img, new_center_x, new_center_y, r


def get_exterior_circle(img, center_x, center_y, r, width=1):
    exterior_r = r + 10  # Exterior circle, to detect measurements /this should be adjusted later.
    # This is NOT the circle being drawn: this is only used to detect measurements.
    output = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # type: mat

    cv2.circle(output, (center_x, center_y), exterior_r, (90, 127, 127), width)

    blue = np.array([90, 127, 127])
    blue2 = np.array([91, 127, 127])
    bluemask = cv2.inRange(output, blue, blue2)  # type: mat

    output = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output[np.where(bluemask == 0)] = 0

    return output


def get_error_margin_circle(img, center_x, center_y, r):
    # We're going for 10 pixels outside and inside the detected circle.
    # It's needed to add a _inside_ circle because sometimes there are some difference between the
    #     detected circle and the "actual" one, this way we try not to lose any measurements.

    output_image = get_exterior_circle(img, center_x, center_y, r, 2 * errorMargin)

    return output_image


def get_red_ink(img):
    input = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red HSV values has limit values for H: both 0 and 180 means "pure red". We apply two masks to keep into
    # account both kinds of red: the more "orangey" and the more "purpley".
    lower_red = np.array([0, 130, 130])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(input, lower_red, upper_red)

    lower_red = np.array([160, 130, 130])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(input, lower_red, upper_red)

    # We join both masks.
    mask = mask0 + mask1

    # Output image will be zero (black) for every pixel except the masked ones (the red-ish ones).
    output = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output[np.where(mask == 0)] = 0
    output[np.where(mask != 0)] = 255

    return output


def identify_groups(path, img):
    kernel = np.ones((10, 10), np.uint8)

    # We apply a morphological filter to reduce noise in the image
    closed_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    boxes = {}

    # Finding shapes in the image
    contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 5 < cv2.contourArea(cnt):  # So we discard rectangles with less than five pixels of area. This reduces noise.
            closest_rect = cv2.minAreaRect(cnt)  # (center(x, y), (width, height), angle of rotation)
            bounding_rect = cv2.boundingRect(cnt)  # (horizontal, vertical, width, height)
            box = cv2.boxPoints(closest_rect)  # (bottom left x and y, and then counterclockwise)
            box = np.int0(box)  # Convert box points to integer
            cv2.drawContours(img, [box], 0, 127, 2)  # This draws the rectangle around the contour
            if path in boxes:
                boxes[path].append((closest_rect, bounding_rect))
            else:
                boxes[path] = [(closest_rect, bounding_rect)]
            util.logger.log(logging.DEBUG, bounding_rect)

    # show_image(img, "With contours")
    return img, boxes


def processPath(path):
    if os.path.isfile(path) is True:
        if "_modified.jpg" not in path:
            readimage(path)
    if os.path.isdir(path) is True:
        file_list = os.listdir(path)
        for file_name in file_list:
            processPath(path + "/" + file_name)


def readimage(path):
    util.logger.log(logging.DEBUG, "Analyzing image at {path}".format(path=path))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    measurements = circles(img, path)
    if measurements > 0:
        util.logger.log(logging.INFO, path + " has measurements!")
    else:
        if measurements == 0:
            util.logger.log(logging.INFO, path + " hasn't measurements!")
        else:
            if measurements == -1:
                util.logger.log(logging.ERROR, "Please review " + path + ", it looks like we see more than one circle.")
            if measurements == -2:
                util.logger.log(logging.ERROR, "Please review " + path + ", it looks like we don't see a circle.")


util.configure_logger()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image or the folder containing images")
args = vars(ap.parse_args())
processPath(args["image"])
