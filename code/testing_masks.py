# Load two images
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


background_image = cv2.imread('../data/background_imgs/bill-wegener-XhX113Jvr5o-unsplash.jpg')
object_image = cv2.imread('../data/extrapolated_objects/1adabf73dd8e489389cc96beec9db6d9.jpg')

x_offset, y_offset = (0, 0)
o_height, o_width = object_image.shape[:2]

"""
lab = cv2.cvtColor(object_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

limg = cv2.merge((cl, a, b))
show_image(limg)
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
show_image(final)

"""

gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)

# edges = cv2.Canny(blur, 10, 100)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# dilated = cv2.dilate(edges, kernel, iterations=1)


thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh_inv = 255 - thresh
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
resulting_contours = max(contours, key=cv2.contourArea)[:, 0, :]

polygon = Polygon(np.array(resulting_contours))

rectangle = tuple(map(int, polygon.bounds))

# mask = np.zeros(object_image.shape[:2], np.uint8)

# backgroundModel = np.zeros((1, 65), np.float64)
# foregroundModel = np.zeros((1, 65), np.float64)

# mask[thresh_inv == 0] = 0
# mask[thresh_inv == 255] = 1
# cv2.grabCut(object_image, mask, rectangle, backgroundModel, foregroundModel, 5, cv2.GC_INIT_WITH_RECT)
# mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# show_image(mask)

# image = object_image * mask[:, :, np.newaxis]
# show_image(image)
# fgbg2 = cv2.createBackgroundSubtractorMOG2()
# fgbg3 = cv2.createBackgroundSubtractorKNN()


# fgmask2 = fgbg2.apply(object_image)
# fgmask3 = fgbg3.apply(object_image)
# show_image(fgmask2)
# show_image(fgmask3)


roi = background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width].copy()
roi_background = cv2.bitwise_and(roi, roi, mask=thresh_inv)

dst = cv2.add(roi_background, object_image)
show_image(dst)

background_image[y_offset: y_offset + o_height, x_offset:x_offset + o_width, :] = dst

show_image(background_image)
