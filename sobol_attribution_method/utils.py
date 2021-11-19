import cv2


def resize(image, shape):
    return cv2.resize(image, shape, interpolation=cv2.INTER_CUBIC)
