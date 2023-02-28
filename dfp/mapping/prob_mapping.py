from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def filter_points(pc, min_h, max_h, ret_idxs=False):
    zs = pc[2, :]
    idxs = np.logical_and(zs > min_h, zs < max_h)
    res = pc[:, idxs]
    return res, idxs if ret_idxs else res


def filter_points_xy_u(pc, x, y, ret_idxs=False):
    xs = np.abs(pc[0, :])
    x_coordinate = pc[0, :]
    ys = np.abs(pc[1, :])
    idxs = np.logical_and(np.logical_and(xs < 2 * x, x_coordinate > 0), ys < y)
    res = pc[:, idxs]
    return res, idxs if ret_idxs else res


def filter_points_xy(pc, x, y, ret_idxs=False):
    xs = np.abs(pc[0, :])
    x_coordinate = pc[0, :]
    ys = np.abs(pc[1, :])
    idxs = np.logical_and(xs < x, ys < y)
    res = pc[:, idxs]
    return res, idxs if ret_idxs else res


class Mapper(object):
    def __init__(self, N=10, world_size=50):
        self.N = N
        self.world_size = world_size
        self.current_map = np.zeros((self.N, self.N), dtype=np.uint8)

    def clear(self):
        self.current_map = np.zeros((self.N, self.N), dtype=np.uint8)

    def add_points(self, pc, pos_x, pos_y, filtered_labels=1):
        pc = np.array(pc)
        pc[0] += self.world_size / 2
        pc[1] += self.world_size / 2
        scaled = pc / self.world_size * (self.N - 1)
        bins = np.round_(scaled).astype(np.uint16)

        if len(bins[1]) == 0:
            return self.current_map

        map = self.current_map[bins[0], bins[1]]
        msk = (map != 255)
        self.current_map[bins[0], bins[1]] = map + msk.astype(np.uint8)

        return self.current_map
