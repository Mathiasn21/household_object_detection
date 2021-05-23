# Load two images
import cv2
import numpy as np
from matplotlib import pyplot as plt



# augs = torchvision.transforms.Compose([
#    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)

"""
def apply(img_tensored_img, aug, shape=(8, 16)):
    rows, columns = shape
    fig_size = 900
    fig = plt.figure(dpi=fig_size)

    for position in range(1, (rows * columns) + 1):
        augmented = aug(img_tensored_img)
        img_altered = augmented.T.numpy() * 255
        img = np.swapaxes(img_altered, 0, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(dtype=np.int32)
        fig.add_subplot(rows, columns, position)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
"""


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def embed_image():
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh_inv = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resulting_contours = max(contours, key=cv2.contourArea)[:, 0, :]

    # polygon = Polygon(np.array(resulting_contours))

    # rectangle = tuple(map(int, polygon.bounds))

    roi = background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width].copy()
    output = cv2.seamlessClone(object_image, background_image, thresh, (o_height + 150, o_width + 150), cv2.NORMAL_CLONE)
    show_image(output)

    roi_background = cv2.bitwise_and(roi, roi, mask=thresh_inv)
    dst = cv2.add(roi_background, object_image)
    background_image[y_offset: y_offset + o_height, x_offset:x_offset + o_width, :] = dst


if __name__ == '__main__':
    background_image = cv2.imread('../data/background_imgs/anthony-tran-iRbHr8jpmGw-unsplash.jpg')
    object_image = cv2.imread('../data/augmented_objects/0a2ca7e9d62c4a96831ad9a245b405c7.jpg')

    x_offset, y_offset = (0, 0)
    o_height, o_width = object_image.shape[:2]
    blur = cv2.GaussianBlur(object_image, (5, 5), 1)
    """
    tran = torchvision.transforms.ToTensor()
    """
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh_inv = 255 - thresh
    # result = poisson_edit(object_image, background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width].copy(), thresh, (0, 0))
    # background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width] = result
    embed_image()

    """
    augmentations = [RandomHorizontalFlip(),
                     RandomVerticalFlip(),
                     RandomRotation((-360, 360)),
                     RandomErasing(scale=(0.02, 0.15), p=0.3),
                     RandomPerspective(distortion_scale=0.65),
                     RandomSolarize(0.4, p=0.2),
                     ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5)]
    aug_transformations = transforms.Compose(augmentations)

    apply(tran(object_image), aug_transformations)
    """

