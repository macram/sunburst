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
    output = img.copy
    gray = grayscaleimage(img)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.imshow("output", np.hstack([img, output]))
        cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
readimage(args["image"])
