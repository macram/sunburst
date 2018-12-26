import numpy as np
import cv2
import argparse
import os


def readimage(path):
    print("Analyzing image at {path}".format(path=path))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    circles(img)


def showimage(img):
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscaleimage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def circles(img):
    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = grayscaleimage(img)

    print("Detecting circles")
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.505, 100, param1=200, param2=150)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            print("Detected circle! Center: ({x},{y}), radius: {r}".format(x=x, y=y, r=r))
            # cv2.circle(output, (x, y), r, (0, 255, 0), 1)
            # cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

        cropped_img, center_x, center_y, r = crop_image(output, x, y, r)
        exterior_circle = get_exterior_circle(cropped_img, center_x, center_y, r)
        red_ink = get_red_ink(cropped_img)
        masked_mask = cv2.bitwise_and(exterior_circle, red_ink)
        if np.count_nonzero(masked_mask) > 0:
            print("We have meditions here!")


def crop_image(img, center_x, center_y, r):
    crop_img = img[center_y - r - 30:center_y + r + 30, center_x - r - 30:center_x + r + 30]
    height, width, channels = crop_img.shape

    new_center_y = height / 2
    new_center_x = width / 2
    print("Center of cropped circle: ({new_center_x},{new_center_y})".format(new_center_x=new_center_x,
                                                                             new_center_y=new_center_y))
    return crop_img, new_center_x, new_center_y, r


def get_exterior_circle(img, center_x, center_y, r):
    exterior_r = r + 10  # Radio del circulo exterior
    output = img.copy()

    cv2.circle(output, (center_x, center_y), exterior_r, (0, 0, 255), 1)

    channels = cv2.split(output)

    ret, exterior_circle = cv2.threshold(channels[2], 250, 255, cv2.THRESH_TOZERO)  # type: (int, mat)

    return exterior_circle


def get_red_ink(img):
    input = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # El rojo en HSV tiene un valor H en los limites: con H valiendo tanto 0 como 180 son rojo "puro". Pillamos
    # dos mascaras precisamente para tomar ambos tipos de rojo: mas "naranja" y mas "violeta".
    lower_red = np.array([0, 30, 30])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(input, lower_red, upper_red)

    lower_red = np.array([160, 30, 30])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(input, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output[np.where(mask == 0)] = 0
    output[np.where(mask != 0)] = 255

    return output


def processPath(path):
    if os.path.isfile(path) is True:
        readimage(path)
    if os.path.isdir(path) is True:
        file_list = os.listdir(path)
        for file_name in file_list:
            processPath(path + "/" + file_name)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
processPath(args["image"])
