##################################################################
# Image utility functions, that are numba optimized.             #
# This is to decrease the latency. Every little bit helps, since #
# there are approx. 71 million total frames processed.           #
##################################################################

import numba as nb
import numpy as np


@nb.jit(nb.uint8[:, :, :](nb.uint8[:, :], nb.uint64), nopython=True)
def one_hot_encode_2d(buf, c):
    """
    One hot encode a 2d array.
    Assumes a 84 x 84 buffer, with 7 different channels to be encoded.
    """
    h, w = buf.shape
    encoded = np.zeros((h, w, c), dtype=np.uint8)
    for i in xrange(buf.shape[0]):
        for j in xrange(buf.shape[1]):
            encoded[i, j, buf[i, j]] = 1

    return encoded


@nb.jit(nb.uint8[:, :, :, :](nb.uint8[:, :, :], nb.uint64), nopython=True)
def one_hot_encode_batch(batch_buf, c):
    """
    One hot encode a 2d array, with some batch size n, Given by (n, 84, 84).
    Assumes the one hot encoded max label is 6. (total of 7 different encodings)
    """
    b, h, w = batch_buf.shape
    encoded = np.zeros((b, h, w, c), dtype=np.uint8)
    for i in xrange(batch_buf.shape[0]):
        encoded[i] = one_hot_encode_2d(batch_buf[i], c)

    return encoded


@nb.jit(nb.uint8[:, :](nb.uint8[:, :, :]), nopython=True)
def rgb2gray(rgb_img):
    grey_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), np.uint8)
    for i in xrange(rgb_img.shape[0]):
        for j in xrange(rgb_img.shape[1]):
            grey_img[i, j] = int(0.2989 * rgb_img[i, j, 0] + 0.5870 *
                                 rgb_img[i, j, 1] + 0.1140 * rgb_img[i, j, 2])

    return grey_img


@nb.jit(nb.uint8[:, :, :](nb.uint8[:, :, :, :]), nopython=True)
def rgb2gray_batch(batch_rgb_imgs):
    grey_imgs = np.zeros((batch_rgb_imgs.shape[0], batch_rgb_imgs.shape[1], batch_rgb_imgs.shape[2]),
                         dtype=np.uint8)
    for i in xrange(batch_rgb_imgs.shape[0]):
        grey_imgs[i] = rgb2gray(batch_rgb_imgs[i])

    return grey_imgs


INTENSITY = 230
BLUE = [INTENSITY, 0, 0]
GREEN = [0, INTENSITY, 0]
RED = [0, 0, INTENSITY]
PURPLE = [INTENSITY, 0, INTENSITY]
YELLOW = [INTENSITY, INTENSITY, 0]
GEGEDROPPING = [193, 157, 103][::-1]
COLORS = [[0] * 3, RED, GREEN, BLUE, PURPLE, YELLOW, GEGEDROPPING]


def color_map(curmap):
    img = np.zeros(list(curmap.shape) + [3])
    for i in range(1, len(COLORS)):
        img[curmap == i] = COLORS[i]
    return img.astype(np.uint8)
