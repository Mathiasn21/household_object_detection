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

# create a simple mask image similar
# to the loaded image, with the
# shape and return type

mask = np.zeros(object_image.shape[:2], np.uint8)

# specify the background and foreground model
# using numpy the array is constructed of 1 row
# and 65 columns, and all array elements are 0
# Data type for the array is np.float64 (default)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

# define the Region of Interest (ROI)
# as the coordinates of the rectangle
# where the values are entered as
# (startingPoint_x, startingPoint_y, width, height)
# these coordinates are according to the input image
# it may vary for different images
polygon = Polygon(np.array(resulting_contours))

rectangle = tuple(map(int, polygon.bounds))

fgbg1 = cv2.BackgroundSubtractor()
fgbg2 = cv2.createBackgroundSubtractorMOG2()
fgbg3 = cv2.createBackgroundSubtractorKNN()


fgmask2 = fgbg2.apply(object_image)
fgmask3 = fgbg3.apply(object_image)
show_image(fgmask2)
show_image(fgmask3)

# mask[thresh_inv == 0] = 0
# mask[thresh_inv == 255] = 1
# apply the grabcut algorithm with appropriate
# values as parameters, number of iterations = 3
# cv2.GC_INIT_WITH_RECT is used because
# of the rectangle mode is used
# cv2.grabCut(object_image, mask, rectangle, backgroundModel, foregroundModel, 5, cv2.GC_INIT_WITH_RECT)

# In the new mask image, pixels will
# be marked with four flags
# four flags denote the background / foreground
# mask is changed, all the 0 and 2 pixels
# are converted to the background
# mask is changed, all the 1 and 3 pixels
# are now the part of the foreground
# the return type is also mentioned,
# this gives us the final mask
# mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# show_image(mask)

# The final mask is multiplied with
# the input image to give the segmented image.
# image = object_image * mask[:, :, np.newaxis]
# show_image(image)

roi = background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width].copy()
roi_background = cv2.bitwise_and(roi, roi, mask=fgmask2)

dst = cv2.add(roi_background, object_image)
show_image(dst)

background_image[y_offset: y_offset + o_height, x_offset:x_offset + o_width, :] = dst

show_image(background_image)
