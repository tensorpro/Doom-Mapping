from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

import numpy as np

import logging
import math

from .net_modules import ImageModule, MeasModule, MapModule
from ..cfg import CFG, VARIABLES

from ..memory.preprocessors import Preprocessors, PreNet
from ..torch_utils.torch_functions import find_max_idx, init_weight


class DFP(nn.Module):
    def __init__(self, experiment_cfg, measurement_dims, action_dims, future_length):
        """
        Construct the DFP Network.
        img_dims           should be a tuple of a single image (CHANNELS, HEIGHT, WIDTH)
        measurement_dims   should be an int of the measurement stream
        action_dims        should be an int of the input action space
        """
        assert isinstance(measurement_dims, int), "measurement_dims must consist of an int"
        assert isinstance(action_dims, int), "action_dims must consist of an int"

        super(DFP, self).__init__()

        # Calculate dependency here
        self.use_img = experiment_cfg.components.save_img
        self.use_calc_automap = experiment_cfg.components.use_calc_automap
        self.use_doom_automap = experiment_cfg.components.use_doom_automap
        # Check for depth
        self.use_gt_depth = experiment_cfg.components.cfg_vision[VARIABLES.depth] == CFG.gt
        self.use_pred_depth = experiment_cfg.components.cfg_vision[VARIABLES.depth] == CFG.pred

        assert not (
            self.use_gt_depth and self.use_pred_depth), 'DFP must use either ground truth or predicted, not both'

        # Check for labels
        self.use_gt_labels = experiment_cfg.components.cfg_vision[VARIABLES.labels] == CFG.gt
        self.use_pred_labels = experiment_cfg.components.cfg_vision[VARIABLES.labels] == CFG.pred

        assert not (
            self.use_pred_labels and self.use_pred_labels), 'DFP must use either ground truth or predicted, not both'

        if measurement_dims == 1:
            obj_coeffs = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
        elif measurement_dims == 3:
            obj_coeffs = np.array([0., 0., 0., 0.25, 0.25, 0.5, 0., 0., 0.,
                                   0.25, 0.25, 0.5, 0., 0., 0., 0.5, 0.5, 1.])
        else:
            raise(NotImplementedError, "Measurement Dimensions must be less than 3, got %d" %
                  (measurement_dims))

        obj_idx = np.where(np.abs(obj_coeffs) > 1e-8)[0]
        obj_coeffs = obj_coeffs[obj_idx]
        self.obj_coeffs = Variable(torch.from_numpy(obj_coeffs[None, :])).float()
        self.obj_idx = Variable(torch.from_numpy(obj_idx)).long()

        if experiment_cfg.use_gpu:
            self.obj_coeffs = self.obj_coeffs.cuda()
            self.obj_idx = self.obj_idx.cuda()

        # Save the dimensions for forward pass use (for reshaping)
        self.future_length = future_length
        self.num_vars = measurement_dims
        self.num_targets = future_length * measurement_dims
        self.num_actions = action_dims

        # Define some aliases sp it's easier to do dependency stuffs
        img_shape = experiment_cfg.img_shape
        map_shape = experiment_cfg.map_shape

        if img_shape is not None:
            logging.info("Initializing DFP Network with image HEIGHT: %d, WIDTH: %d, CHANNELS: %d" %
                         (img_shape[1], img_shape[2], img_shape[0]))
        if map_shape is not None:
            logging.info("Initializing DFP Network with image HEIGHT: %d, WIDTH: %d, CHANNELS: %d" %
                         (map_shape[1], map_shape[2], map_shape[0]))

        logging.info("Network recognizes %d actions and %d targets" %
                     (self.num_actions, self.num_targets))

        #################
        # Network Setup #
        #################

        # First set up the corresponding vectors
        if img_shape is not None:
            self.pre_img = ImageModule(img_shape)
        else:
            self.pre_img = None

        self.pre_meas = MeasModule(measurement_dims)

        # Then setup the mapping
        if map_shape is not None:
            self.pre_automap = MapModule(map_shape)
        else:
            self.pre_automap = None

        # resolve
        if img_shape is not None and map_shape is not None:
            j_size = (list(self.pre_img.parameters())[-1].size()[0] +
                      list(self.pre_meas.parameters())[-1].size()[0] +
                      list(self.pre_automap.parameters())[-1].size()[0])
        elif img_shape is not None:
            j_size = (list(self.pre_img.parameters())[-1].size()[0] +
                      list(self.pre_meas.parameters())[-1].size()[0])
        elif map_shape is not None:
            j_size = (list(self.pre_automap.parameters())[-1].size()[0] +
                      list(self.pre_meas.parameters())[-1].size()[0])

        else:
            raise NotImplementedError, "Either automap or vision must be used"

        # VAL predicts the means
        self.val_fc = nn.Sequential(
            nn.Linear(j_size, 512, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.num_targets, bias=True)
        )

        self.adv_fc = nn.Sequential(
            nn.Linear(j_size, 512, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.num_actions * self.num_targets, bias=True)
        )

        # Initialize everything
        init_weight(self)

    def forward(self, image, measurements, automap, actions=None, objectives=None):
        # Complete image forward pass
        accum_vec = []
        if self.pre_img is not None:
            accum_vec.append(self.pre_img(image))
        accum_vec.append(self.pre_meas(measurements))
        if self.pre_automap is not None:
            accum_vec.append(self.pre_automap(automap))

        # Stack
        j = torch.cat(accum_vec, 1)

        val = self.val_fc(j).view(-1, 1, self.num_targets)
        adv = self.adv_fc(j).view(-1, self.num_actions, self.num_targets)
        pred = adv - adv.mean(1).unsqueeze(1)
        pred = pred + val

        return pred

    def apply_inference(self, images, depths, labels, measurements,
                        automap, automap_labels, CUDA=False, preprocess=True):
        if preprocess:
            images = Preprocessors.preprocess_images(images, depths, labels)
            measurements = Preprocessors.preprocess_measurements(measurements)
            automap = Preprocessors.preprocess_automaps(automap, automap_labels)

        # Images
        if self.pre_img is not None:
            if len(images.shape) == 2:
                images = torch.from_numpy(images).unsqueeze(0).unsqueeze(0)
            elif len(images.shape) == 3:
                images = torch.from_numpy(images).unsqueeze(1)
            elif len(images.shape) == 4:
                images = torch.from_numpy(images)

            images = Variable(images, volatile=True).float()

            if CUDA:
                images = images.cuda()

        # Measurements
        if len(measurements.shape) == 1:
            measurements = torch.from_numpy(measurements).unsqueeze(0)
        else:
            measurements = torch.from_numpy(measurements)
        measurements = Variable(measurements, volatile=True).float()

        if CUDA:
            measurements = measurements.cuda()

        # Automaps
        if self.pre_automap is not None:
            if len(automap.shape) == 2:
                automap = torch.from_numpy(automap).unsqueeze(0).unsqueeze(0)
            elif len(automap.shape) == 3:
                automap = torch.from_numpy(automap).unsqueeze(1)
            elif len(automap.shape) == 4:
                automap = torch.from_numpy(automap)
            automap = Variable(automap, volatile=True).float()

            if CUDA:
                automap = automap.cuda()

        result = self(images, measurements, automap=automap)
        curr_preds, curr_obj, curr_act = find_max_idx(result, self.obj_coeffs,
                                                      self.obj_idx,
                                                      NUMPY=True)

        return curr_act, curr_preds
