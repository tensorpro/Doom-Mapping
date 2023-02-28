import numpy as np
import cv2


def downsize(img, factor, interpolation=cv2.INTER_NEAREST):
    a, b = img.shape[:2]
    return cv2.resize(img, (b // factor, a // factor), interpolation=interpolation)


def point_cloud(depth, ds=1):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    cx = 80
    cy = 60
    fx = 80
    fy = 96
    Rbc = np.zeros((3, 3))
    Rbc[0, 2] = 1
    Rbc[1, 0] = 1
    Rbc[2, 1] = 1
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(0, cols, ds),
                       np.arange(0, rows, ds), sparse=True)
    depth = cv2.resize(depth, (len(c[0]), len(r)), interpolation=cv2.INTER_NEAREST)
    z = depth
    x = z * (c - cx) / fx
    y = z * (r - cy) / fy
    pc = np.dstack((x, y, z))
    pc = pc.transpose(2, 0, 1).reshape(3, -1)
    pc = np.dot(Rbc, pc)
    return pc


def _add_points(depth, current_map, world_size=50, N=10, lab=None):
    pc = point_cloud(depth)
    if lab is 1:
        return current_map
    if lab is None:
        lab = 1
        return current_map
    # lab[lab==1]=6
    pc = np.array(pc) + world_size / 2
    pc = np.clip(pc, 0, world_size - 2)
    scaled = pc / world_size * (N)
    bins = np.round_(scaled).astype(np.uint16)
    idx = np.where(lab)
    bins = bins[:, idx]
    lab = lab[idx]
    current_map[bins[0], bins[1]] = lab

    return current_map


def add_points(self, depth, lab=None):
    l = _add_points(depth, self.current_map, self.world_size, self.N, lab)
    return l, l


class Map:
    # add_points = add_points

    def __init__(self, N=42, world_size=72, downsample_rate=4, match=True):
        self.N = N
        self.world_size = world_size
        self.current_map = np.zeros((N, N))
        self.downsample_rate = downsample_rate
        self.match = match

    def add_points(self, depth, labels=None):
        self.current_map[:] = 0
        # Check to see if the current mapper returned a faulty state (reset)
        if depth is None:
            self.current_map = np.zeros((self.N, self.N))
            return self.current_map

        world_size = self.world_size
        N = self.N
        pc = point_cloud(depth, self.downsample_rate)
        pc[1] += world_size / 2
        scaled = pc[:2] / world_size * (N)
        bins = np.clip(np.round_(scaled), 0, N - 1).astype(np.uint16)
        if labels is not None:
            labels = downsize(labels, self.downsample_rate).flatten()
            if self.match:
                idx = np.where(labels)
                bins = np.squeeze(bins[:, idx])
                fill_val = labels[idx]
            else:
                fill_val = labels
        else:
            fill_val = 1
        self.current_map[bins[0], bins[1]] = fill_val
        return self.current_map

    def show(self, size=(200, 200)):
        return cv2.resize(color_map(self.current_map), size,
                          interpolation=cv2.INTER_NEAREST)
