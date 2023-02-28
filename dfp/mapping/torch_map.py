import numpy as np
import torch
import cv2
from ..img_utils.utils import color_map

REPL_IDX = 6


def point_cloudb(depth, c, r, ds=1, mode=torch):
    cx = 80
    cy = 60
    fx = 80
    fy = 96
    B, rows, cols = depth.size()

    z = depth
    x = z * (c - cx).expand_as(z) / fx
    y = z * (r - cy).expand_as(z) / fy
    h = torch.ones(x.size()).type(mode.FloatTensor)
    pc = torch.stack((-x, -y, z, h), 1)
    pc = pc.contiguous().view(B, 4, -1)
    pc = torch.index_select(pc, 1, mode.LongTensor([2, 0, 1, 3]))
    return pc


def rotb(angles, mode=torch):
    B = len(angles)
    s = torch.sin(angles).type(mode.FloatTensor)
    c = torch.cos(angles).type(mode.FloatTensor)
    zs = torch.zeros(s.size()).type(mode.FloatTensor)
    os = torch.ones(s.size()).type(mode.FloatTensor)
    tfm = torch.stack([c, s, zs,
                       -s, c, zs,
                       zs, zs, os], 1).view(B, 3, 3)
    return tfm


def get_tfm(p0, p1, a0, a1, mode=torch):
    B = len(p0)
    tfm = torch.zeros(B, 4, 4).type(mode.FloatTensor)
    R10 = rotb(a1 - a0, mode)
    tfm[:, :3, :3] = R10
    R1 = torch.bmm(rotb(a1, mode), (p0 - p1).view(B, 3, 1))
    tfm[:, :3, 3] = R1
    tfm[:, -1, -1] = 1
    return tfm


class History:

    def __init__(self, B, H, pcsize, mode=torch):
        self.B = B
        self.H = H
        self.pcsize = pcsize
        self.points = torch.zeros(B, 4, pcsize * H).type(mode.FloatTensor)
        self.labels = torch.zeros(B, pcsize * H).type(mode.FloatTensor)
        self.i = 0
        self.pos = torch.zeros(B, 3).type(mode.FloatTensor)
        self.ang = torch.zeros(B).type(mode.FloatTensor)
        self.mode = mode

    def add(self, pc, pos, ang, labels=1, to_clear=None):
        mode = self.mode
        tfms = get_tfm(self.pos, pos, self.ang, ang, mode=mode)

        # NOTE: Leave dropping wherever the agent visited
        curr_pos = torch.zeros(pc.shape[:-1])
        curr_pos[:, -1] = 1

        pc[:, :, 0] = curr_pos
        labels[:, 0] = REPL_IDX

        self.points = torch.bmm(tfms, self.points)
        retpc = torch.cat((self.points, pc), 2)
        retl = torch.cat((self.labels, labels), 1)
        if to_clear is not None:
            labels[to_clear] = 0
        start = self.i * self.pcsize
        end = start + self.pcsize
        self.points[:, :, start: end] = pc
        self.pos = pos
        self.ang = ang
        self.i = (self.i + 1) % self.H
        self.labels[:, start: end] = labels
        return retpc, retl


class MultiMap:
    # add_points = add_points

    def __init__(self, N=84, world_size=72 / 14, B=8, H=16,
                 downsample_rate=4, mode=torch):
        self.N = N
        self.B = B
        self.H = H
        self.world_size = world_size
        self.current_map = torch.zeros((B, N, N)).type(mode.FloatTensor)
        self.downsample_rate = downsample_rate

        self.hist = History(B, H, 30 * 40, mode=mode)

        rows, cols = 120, 160
        ds = downsample_rate
        c, r = np.meshgrid(np.arange(0, cols, ds),
                           np.arange(0, rows, ds), sparse=True)
        self.c = torch.from_numpy(c.astype(np.float32)).type(mode.FloatTensor)
        self.r = torch.from_numpy(r.astype(np.float32)).type(mode.FloatTensor)
        self.mode = mode
        self.b = torch.from_numpy(
            np.repeat(np.arange(self.B), 30 * 40 * (H + 1))).type(mode.LongTensor)

    def add_points(self, depth, labels, pos, angle, term=None):
        world_size = self.world_size
        N = self.N
        self.current_map[:] = 0
        mode = self.mode
        pos = torch.from_numpy(np.array(pos)).type(mode.FloatTensor) / 100.
        angle = torch.from_numpy(np.array(angle)).type(mode.FloatTensor) / 57.3
        depth = torch.from_numpy(np.array(depth)).type(mode.FloatTensor) / 14.
        labels = torch.from_numpy(np.array(labels)).type(mode.FloatTensor)
        depth = depth[:, ::self.downsample_rate, ::self.downsample_rate]

        labels = labels[:, ::self.downsample_rate,
                        ::self.downsample_rate].contiguous().view(self.B, -1)
        pc = point_cloudb(depth, self.c, self.r, self.downsample_rate, mode=mode)
        labels = labels.contiguous().view(self.B, -1)
        to_clear = (labels == 1) | (depth.contiguous().view(self.B, -1) > 200 / 14)
        pc, labels = self.hist.add(pc, pos, angle, labels, to_clear)

        pc[:, :2] += world_size / 2
        scaled = pc[:, :2] / world_size * (N)
        bins = torch.clamp(scaled, 0, N - 1).round_().long()
        idx = labels > 0
        labels = labels[idx]
        x = bins[:, 0, :].contiguous().view(-1)
        y = bins[:, 1, :].contiguous().view(-1)
        b = self.b

        if b.dim() > 0:
            self.current_map[b[idx], x[idx], y[idx]] = labels

        return self.current_map.cpu().numpy()
        # cmap = self.current_map.cpu().numpy()
        # colored = color_map(cv2.resize(cmap[0], (400, 400),
        #                                interpolation=cv2.INTER_NEAREST))
        # cv2.imshow("w/ GEGE droppings", colored)

        # # cmap[cmap == REPL_IDX] = 0
        # return self.current_map.cpu().numpy()
