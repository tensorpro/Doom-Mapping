from __future__ import division
import numpy as np
import math
from transforms import transformations
import matplotlib.pyplot as plt
import cv2
import time

from ..img_utils.utils import color_map


class Mapper(object):
    def __init__(self, N=10, world_size=50):
        self.N = N
        self.world_size = world_size
        self.current_map = np.zeros((self.N, self.N), dtype=np.uint8)
        self.count = 1

    def clear(self):
        self.count += 1

    def add_points(self, pc, pos_x, pos_y):
        pc = np.array(pc)
        pc[0] += self.world_size / 2
        pc[1] += self.world_size / 2
        scaled = pc / self.world_size * (self.N - 1)
        bins = np.round_(scaled).astype(np.uint16)

        if len(bins[1]) == 0:
            return self.current_map

        self.current_map[bins[0], bins[1]] = self.count
        return self.current_map


class VDMapper(object):
    """
    Generate a probabilitic map for vizdoom
    """

    def __init__(self, local_map_sz=10, local_disp_sz=100, global_map_sz=50,
                 global_disp_sz=800, N=10, world_size=50):

        self.global_map = Mapper(global_disp_sz, global_map_sz)
        # self.global_map = Mapper(global_disp_sz, world_size)

        self.Rbc = np.zeros((3, 3))
        self.Rbc[0, 2] = 1
        self.Rbc[1, 0] = 1
        self.Rbc[2, 1] = 1

        self.Rcb = np.transpose(self.Rbc)

        self.cx = 80
        self.cy = 60
        self.fx = 80
        self.fy = 96

        self.local_map_sz = local_map_sz
        self.local_disp_sz = local_disp_sz

        self.global_map_sz = global_map_sz
        self.global_disp_sz = global_disp_sz

        self.crop_sz = 80

        self.frames = 0

    def init_rot_mtx(self, yaw_origin):
        """
        Generate the rotation matrix for the yaw.
        """
        self.rot_mtx = np.asarray([[math.cos(yaw_origin / 57.3),
                                    math.sin(yaw_origin / 57.3),
                                    0],
                                   [-math.sin(yaw_origin / 57.3),
                                    math.cos(yaw_origin / 57.3),
                                    0],
                                   [0, 0, 1]])

    def point_cloud_gt(self, depth, downsample=1):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.

        """
        cx = self.cx
        cy = self.cy
        fx = self.fx
        fy = self.fy
        rows, cols = depth.shape

        c, r = np.meshgrid(np.arange(0, cols, downsample),
                           np.arange(0, rows, downsample), sparse=True)
        depth = depth[::downsample, ::downsample]
        valid = (depth > 0) & (depth < 140)
        z = np.where(valid, depth / 14.0, np.nan)
        x = np.where(valid, z * (c - cx) / fx, 0)
        y = np.where(valid, z * (r - cy) / fy, 0)

        pc = np.dstack((x, y, z))
        return pc

    # probabilistic map (we don't consider the labels for now)
    def compute_global_map_prob(self, depth, coord, term):
        if term:
            f = np.random.rand(1)[0]
            # TODO: is this really correct
            self.frames = 0
            self.global_map.clear()
            # return a blank if terminal is reached
            return (np.zeros((self.global_map.N, self.global_map.N), dtype=np.uint8),
                    np.zeros((self.global_map.N, self.global_map.N), dtype=np.uint8))

        coord[:3] = coord[:3] / 100.0

        if self.frames == 0:
            self.org_x = coord[0]
            self.org_y = coord[1]
            self.org_z = coord[2]

            self.init_rot_mtx(0)

        # coord[0] = coord[0] - self.org_x
        # coord[1] = coord[1] - self.org_y
        # coord[2] = coord[2] - self.org_z
        L = -5.5
        coord[0] = coord[0] + L
        coord[1] = coord[1] + L
        # coord[2] = coord[2] - self.org_z

        coord[:3] = np.dot(self.rot_mtx, coord[:3])
        coord[1] = -coord[1]

        pc = self.point_cloud_gt(depth, 4)

        pc_b = np.dot(self.Rbc, pc.transpose(2, 0, 1).reshape(3, -1))
        pc_b = np.array([[0, 0, 0]]).T

        Rwb = transformations.euler_matrix(0, 0, coord[3] / 57.3, axes='sxyz')
        Rwb = Rwb[0:3, 0:3]
        dotted = np.dot(Rwb, pc_b)
        pc_w = np.expand_dims(coord[:-1], 0).T + dotted

        if pc_b.shape[1] != 0:
            np.clip(pc_w[:2], -self.global_map_sz // 2 + 1,
                    self.global_map_sz // 2 - 1, pc_w[:2])

            self.global_map.add_points(pc_w, coord[0], coord[1])

        self.frames += 1
        # self.show()
        return self.global_map.current_map

    def show(self):
        _map = self.global_map.current_map
        _map = cv2.resize(_map, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("color map", color_map(_map))
        cv2.waitKey(1)
