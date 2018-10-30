import numpy as np
import cv2


def readimage(file="/Users/manu/photo_2018-06-24_21-03-25.jpg"):
    img = cv2.imread(file, 0)

    cv2.imshow(file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


readimage()
