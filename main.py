import numpy as np
import cv2
import argparse


def readimage(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    circles(img)


def showimage(img):
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscaleimage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def circles(img):
    output = img.copy()
    gray = grayscaleimage(img)

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
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)
            cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

        showimage(exterior_circle(output, x, y, r))


def crop_image(img, center_x, center_y, r):
    crop_img = img[center_y - r - 30:center_y + r + 30, center_x - r - 30:center_x + r + 30]
    return crop_img


def exterior_circle(img, center_x, center_y, r):
    exterior_r = r + 10  # Radio del circulo exterior
    output = img.copy()
    gray = grayscaleimage(img)

    cv2.circle(output, (center_x, center_y), exterior_r, (255, 0, 0), 1)

    return crop_image(output, center_x, center_y, r)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
readimage(args["image"])
