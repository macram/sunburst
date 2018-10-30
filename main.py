import numpy as np
import cv2


def readimage(path="/Users/manu/photo_2018-06-24_21-03-25.jpg"):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    showimage(greyscaleimage(img))


def showimage(img):
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def greyscaleimage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


readimage()
