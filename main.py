import numpy as np
import cv2
import argparse
import os


def readimage(path):
    # print("Analyzing image at {path}".format(path=path))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    meditions = circles(img, path)
    if meditions > 0:
        print(path + " has meditions!")
    else:
        if meditions == 0:
            print(path + " hasn't meditions!")
        else:
            if meditions == -1:
                print("Please review " + path + ", it looks like we see more than one circle.")
            if meditions == -2:
                print("Please review " + path + ", it looks like we don't see a circle.")


def show_image(img, title="Imagen"):
    return
    # cv2.imshow(title, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def save_image(img, path, suffix=""):
    cv2.imwrite(path + suffix + "_modified.jpg", img)


def grayscale_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def circles(img, path=""):
    output = img.copy()
    gray = grayscale_image(img)

    # print("Detecting circles")
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.505, 100, param1=400, param2=150)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        intcircles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in intcircles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            # print("Detected circle! Center: ({c_x},{c_y}), radius: {c_r}".format(c_x=x, c_y=y, c_r=r))
            cv2.circle(output, (x, y), r, (0, 255, 255), 1)
            # cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

        cropped_img, center_x, center_y, r = crop_image(output, x, y, r)
        save_image(cropped_img, path, "_detectedcircle")

        if intcircles.size == 3:
            exterior_circle = get_exterior_circle(cropped_img, center_x, center_y, r)
            red_ink = get_red_ink(cropped_img)
            masked_mask = cv2.bitwise_and(exterior_circle, red_ink)
            # showimage(cropped_img, path)
            save_image(exterior_circle, path, "_exteriorcircle")
            save_image(red_ink, path, "_redink")
            save_image(masked_mask, path, "_maskedmask")
            with_groups = identify_groups(red_ink)
            save_image(with_groups, path, "_withgroups")
            if np.count_nonzero(masked_mask) > 0:
                return 1
            else:
                return 0
        else:
            return -1
    else:
        return -2


def crop_image(img, center_x, center_y, r):
    crop_img = img[center_y - r - 30:center_y + r + 30, center_x - r - 30:center_x + r + 30]
    height, width, channels = crop_img.shape

    new_center_y = height / 2
    new_center_x = width / 2
    # print("Center of cropped circle: ({new_center_x},{new_center_y})".format(new_center_x=new_center_x,
    #                                                                         new_center_y=new_center_y))
    return crop_img, new_center_x, new_center_y, r


def get_exterior_circle(img, center_x, center_y, r):
    exterior_r = r + 10  # Radio del circulo exterior
    output = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # type: mat

    cv2.circle(output, (center_x, center_y), exterior_r, (90, 127, 127), 1)

    blue = np.array([90, 127, 127])
    blue2 = np.array([91, 127, 127])
    bluemask = cv2.inRange(output, blue, blue2)  # type: mat

    output = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output[np.where(bluemask == 0)] = 0

    return output


def get_red_ink(img):
    input = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # El rojo en HSV tiene un valor H en los limites: con H valiendo tanto 0 como 180 son rojo "puro". Pillamos
    # dos mascaras precisamente para tomar ambos tipos de rojo: mas "naranja" y mas "violeta".
    lower_red = np.array([0, 130, 130])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(input, lower_red, upper_red)

    lower_red = np.array([160, 130, 130])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(input, lower_red, upper_red)

    # Unir ambas mascaras
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output[np.where(mask == 0)] = 0
    output[np.where(mask != 0)] = 255

    return output


def identify_groups(img):
    kernel = np.ones((10, 10), np.uint8)

    closed_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    image, contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 5 < cv2.contourArea(cnt):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, 127, 2)
            print(cv2.boundingRect(cnt))

    show_image(img, "With contours")
    return img


def processPath(path):
    if os.path.isfile(path) is True:
        if "_modified.jpg" not in path:
            readimage(path)
    if os.path.isdir(path) is True:
        file_list = os.listdir(path)
        for file_name in file_list:
            processPath(path + "/" + file_name)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image or the folder containing images")
args = vars(ap.parse_args())
processPath(args["image"])
