import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

import logging


def find_max_idx(preds, obj, obj_idx, NUMPY=False):
    """
    goal should be a goal vector (with the number of variables)
    preds should be the raw output from the DFP prediction network
    """
    # Resize the goal to match dims
    curr_preds = preds.index_select(2, obj_idx) * obj

    curr_obj = torch.sum(curr_preds, 2)
    curr_act = torch.max(curr_obj, 1)[1]

    if NUMPY:
        return curr_preds.data.cpu().numpy(), curr_obj.data.cpu().numpy(), curr_act.data.cpu().numpy()
    else:
        return curr_preds, curr_obj, curr_act


def to_super_sparse(idx, actions=8, vars=6):
    act = np.zeros((actions, vars))
    act[idx] = np.ones(vars)

    return act


def to_sparse(idx, actions=256):
    idx = int(idx)
    act = np.zeros(actions)
    act[idx] = 1

    return act


def loss(preds, actions, f, CUDA=False):
    """
    Parameterized L2 loss, ignoring nans
    Uses the goal vector to filter the output
    """
    num_actions = preds.size()[1]
    num_vars = preds.size()[2]

    # obtain the predictions, maybe we can vectorize this
    idx = Variable(torch.Tensor(actions)).byte()

    if CUDA:
        idx = idx.cuda()

    idx = idx.unsqueeze(2).expand_as(preds)

    preds = preds.masked_select(idx)
    preds = preds.view(-1, num_vars)

    # Apparently torch doesn't have an easy way to index NaNs, so here's the hack
    nan_mask = f.ne(f)
    f[nan_mask] = preds[nan_mask]

    # Original DFP uses a sum, I don't like it, but leave it as a sum for now.

    return torch.sum(torch.pow(f - preds, 2).mean(0))


def adjust_learning_rate(optimizer, lr, decay_rate, decay_steps, curr_steps, steps=True):
    if steps:
        new_lr = lr * math.pow(decay_rate, int(curr_steps / decay_steps))
    else:
        new_lr = lr * math.pow(decay_rate, float(curr_steps) / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr  # return for logging purposes


def calculate_eps(step):
    return 0.02 + 145000. / (float(step) + 150000.)


def calculate_out_padding(in_dim, kernel_size, padding=1, dilation=1, stride=1):
    return math.floor((in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def print_stats(vect, name):
    logging.info('\t%s: \n\t\tAverage: %10.10f\n\t\tMedian: %10.10f\n\t\tMin: %10.10f\n\t\tMax: %10.10f\n\t\tTotal: %10.10f' %
                 (name, vect.mean(), np.median(vect), vect.min(), vect.max(), vect.sum()))


def calc_stddev(c_o, k_h, k_w):
    """
    Calculate the standard deviation required for intiaizing the network
    """
    return 2.0 / np.sqrt(c_o * k_h * k_w)


def init_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            stddev = 0.9 * math.sqrt(2.0 / n)
            # truncated_normal doesn't exist, simulate via modulus
            m.weight.data = torch.fmod(
                m.weight.data.normal_(0, stddev), 2 * stddev)

            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.in_features
            stddev = 0.9 * math.sqrt(2.0 / n)
            m.weight.data = torch.fmod(
                m.weight.data.normal_(0, stddev), 2 * stddev)

            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Sequential):
            for mini_module in list(m.modules())[1:]:
                init_weight(mini_module)


def calc_padding(in_height, in_width,
                 filter_height, filter_width,
                 stride):

    out_height = math.ceil(float(in_height) / float(stride))
    out_width = math.ceil(float(in_width) / float(stride))

    pad_along_height = max((out_height - 1) * stride +
                           filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * stride +
                          filter_width - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (out_height, out_width, pad_top, pad_bottom, pad_left, pad_right)
