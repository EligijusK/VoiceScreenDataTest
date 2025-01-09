import cv2


def resize_image(img, out_size):
    h, w = img.shape[0:2]
    aspect = w / h

    img = cv2.resize(img, (out_size, int(out_size / aspect)), interpolation=cv2.INTER_NEAREST)

    return img
