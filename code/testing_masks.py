# Load two images
import cv2
import numpy as np
from shapely.geometry import Polygon


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


background_image = cv2.imread('../data/background_imgs/bill-wegener-XhX113Jvr5o-unsplash.jpg')
object_image = cv2.imread('../data/extrapolated_objects/0b9f35cba6874c3c995095b2d3f84181.jpg')

x_offset, y_offset = (0, 0)
o_height, o_width = object_image.shape[:2]

gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)

out = cv2.addWeighted(object_image, 80, object_image, 0, 50)
show_image(out)


thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh_inv = 255 - thresh
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
resulting_contours = max(contours, key=cv2.contourArea)[:, 0, :]

polygon = Polygon(np.array(resulting_contours))

rectangle = tuple(map(int, polygon.bounds))

roi = background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width].copy()
roi_background = cv2.bitwise_and(roi, roi, mask=thresh_inv)

dst = cv2.add(roi_background, object_image)
show_image(dst)

background_image[y_offset: y_offset + o_height, x_offset:x_offset + o_width, :] = dst

show_image(background_image)
