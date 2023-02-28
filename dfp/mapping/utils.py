"""
Highly optimized utility functions for vizdoom_pointcloud.py
"""
from __future__ import print_function
import numba as nb
import numpy as np


@nb.jit(nb.float64[:, :](nb.boolean[:, :], nb.float64[:, :], nb.float64), nopython=True)
def nb_where(valid_mask, x, y):
    arr = np.zeros(valid_mask.shape, dtype=np.float64)
    col, row = valid_mask.shape
    for i in xrange(col):
        for j in xrange(row):
            if valid_mask[i, j]:
                arr[i, j] = x[i, j]
            else:
                arr[i, j] = y

    return arr


@nb.jit(nb.float64[:, :, :](nb.int64, nb.int64, nb.int64, nb.int64,
                            nb.uint8[:, :], nb.int64[:, :], nb.int64[:, :]), nopython=True)
def point_cloud(cx, cy, fx, fy, depth, c, r):
    valid = (depth > 0) & (depth < 255)
    # print(valid.shape)
    z = nb_where(valid, depth / 14.0, np.nan)
    x = nb_where(valid, z * (c - cx) / fx, 0)
    y = nb_where(valid, z * (r - cy) / fy, 0)

    pc = np.dstack((x, y, z))
    return pc


@nb.jit(nb.types.Tuple((nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.int64, nb.int64), nopython=True)
def filter_points_xy(pc, x, y):
    xs = np.abs(pc[0, :])
    x_coordinate = pc[0, :]
    ys = np.abs(pc[1, :])
    pcT = pc.T
    idxs = np.logical_and(xs < x, ys < y)
    res = pcT[idxs].T
    return res, idxs


@nb.jit(nb.types.Tuple((nb.float64[:, :], nb.boolean[:]))(nb.float64[:, :], nb.float64, nb.float64), nopython=True)
def filter_points(pc, min_h, max_h):
    zs = pc[2, :]
    pcT = pc.T
    idxs = np.logical_and(zs > min_h, zs < max_h)
    res = pcT[idxs].T
    return res, idxs


@nb.jit(nb.uint32[:](nb.float64[:]), nopython=True)
def nb_round(arr1d):
    """
    In place rounding for 1d arrays. Assumes positive floats
    """
    for i in xrange(arr1d.shape[0]):
        rem = arr1d[i] - int(arr1d[i])
        if rem > 0.5:
            arr1d[i] = np.floor(arr1d[i]) + 1
        else:
            arr1d[i] = np.floor(arr1d[i])

    return arr1d.astype(np.uint32)


@nb.jit(nb.uint8[:, :](nb.uint8[:, :], nb.uint16[:], nb.uint16[:], nb.uint8), nopython=True)
def assoc2d_scalar(arr, bin1, bin2, label):
    """
    Equivalent to arr[bin1, bin2] = label, when label is a scalar value.
    Assumes that bin1 and bin2 are the same size
    """
    for elm in xrange(bin1.shape[0]):
        arr[bin1[elm], bin2[elm]] = label

    return arr


@nb.jit(nb.uint8[:, :](nb.uint8[:, :], nb.uint16[:], nb.uint16[:], nb.uint8[:]), nopython=True)
def assoc2d_1d(arr, bin1, bin2, label):
    """
    Equivalent to arr[bin1, bin2] = label, when label is a 1d array
    Assumes that labels is the same size as bin1 and bin2
    """
    for elm in xrange(bin1.shape[0]):
        arr[bin1[elm], bin2[elm]] = label[elm]

    return arr


@nb.jit(nb.uint8[:, :](nb.float64[:, :], nb.uint8[:, :], nb.int64, nb.int64, nb.int64), nopython=True)
def add_points_scalar(pc, current_map, labels, world_size, N):
    """
    Add points when label is a scalar value. 
    """
    pc_x = pc[0] + world_size / 2
    pc_y = pc[1] + world_size / 2
    scaled_x = pc_x / world_size * (N - 1)
    scaled_y = pc_y / world_size * (N - 1)
    bins_x = nb_round(scaled_x).astype(np.uint16)
    bins_y = nb_round(scaled_y).astype(np.uint16)

    return assoc2d_scalar(current_map, bins_x, bins_y, np.uint8(labels))


@nb.jit(nb.uint8[:, :](nb.float64[:, :], nb.uint8[:, :], nb.uint8[:], nb.int64, nb.int64), nopython=True)
def add_points_1d(pc, current_map, labels, world_size, N):
    """
    Add points when label is 1 dimensional
    """
    pc_x = pc[0] + world_size / 2
    pc_y = pc[1] + world_size / 2
    scaled_x = pc_x / world_size * (N - 1)
    scaled_y = pc_y / world_size * (N - 1)
    bins_x = nb_round(scaled_x).astype(np.uint16)
    bins_y = nb_round(scaled_y).astype(np.uint16)

    return assoc2d_1d(current_map, bins_x, bins_y, labels.astype(np.uint8))
