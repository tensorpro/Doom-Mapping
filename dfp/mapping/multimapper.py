####################################################################################
# Wrapper to help multiprocess mapping.                                            #
# Unfortunately, some module level ugliness is required to make the implementation #
# at least seem a tiny bit cleaner.                                                #
####################################################################################


import numpy as np
import logging
import cv2

from ..cfg import VARIABLES, CFG

from .map import Map
from ..img_utils.utils import color_map
from torch_map import MultiMap
import torch


class MultiMapper(object):
    def __init__(self, experiment_cfg):
        num_mappers = experiment_cfg.num_agents
        logging.info("Starting %d mappers" % (num_mappers))
        self.num_mappers = num_mappers
        self.map_resolution = experiment_cfg.map_shape

        map_comp = experiment_cfg.components.cfg_automap
        map_label_comp = experiment_cfg.components.cfg_automap_labels
        self.use_gt_depth = (map_comp[VARIABLES.depth] == CFG.gt or
                             map_label_comp[VARIABLES.depth] == CFG.gt)

        self.use_gt_labels = (map_label_comp[VARIABLES.labels] == CFG.gt or
                              map_label_comp[VARIABLES.labels] == CFG.gt_enemies)

        self.use_gt_motion = (map_comp[VARIABLES.motion] == CFG.gt or
                              map_label_comp[VARIABLES.motion] == CFG.gt)

        self.use_pred_depth = (map_comp[VARIABLES.depth] == CFG.pred or
                               map_label_comp[VARIABLES.depth] == CFG.pred)

        self.use_pred_labels = (map_label_comp[VARIABLES.labels] == CFG.pred or
                                map_label_comp[VARIABLES.labels] == CFG.pred_enemies)

        self.use_pred_motion = (map_comp[VARIABLES.motion] == CFG.pred or
                                map_label_comp[VARIABLES.motion] == CFG.pred)

        mode = torch.cuda if experiment_cfg.use_gpu else torch
        self.tmapper = MultiMap(N=experiment_cfg.map_shape[1],
                                B=experiment_cfg.num_agents,
                                H=experiment_cfg.map_history,
                                mode=mode)

    def reset(self):
        self.tmapper.hist.labels *= 0
        self.tmapper.hist.points *= 0

    def __call__(self, batch_depth, batch_labels, batch_coord, batch_term):
        """
        call the mappers with appropriate arguments
        batch_coord is expected to be x, y, z, angle
        """
        if batch_depth.shape != (120, 160):
            batch_depth = [cv2.resize(img[0], (160, 120),
                                      interpolation=cv2.INTER_NEAREST)
                           for img in batch_depth]

        if batch_labels is not None and batch_labels.shape != (120, 160):
            batch_labels = [cv2.resize(img[0], (160, 120),
                                       interpolation=cv2.INTER_NEAREST)
                            for img in batch_labels]

        batch_pos = batch_coord[:, :3]
        batch_ang = batch_coord[:, 3]

        should_clear = (1 - torch.FloatTensor(batch_term.astype(np.float32))
                        ).view(self.tmapper.B, 1)
        should_clear = should_clear.type(self.tmapper.mode.FloatTensor)
        self.tmapper.hist.labels *= should_clear
        self.tmapper.hist.points *= should_clear.view(self.tmapper.B, 1, 1)
        label_map = self.tmapper.add_points(batch_depth,
                                            batch_labels,
                                            batch_pos,
                                            batch_ang,
                                            batch_term)

        label_map = np.expand_dims(label_map, axis=1).astype(np.uint8)

        # for i, am in enumerate(automaps):
        #     cv2.imshow(str(i), cv2.resize(color_map(am[0]), (200, 200)))
        #     cv2.imshow(str(i) + 'd', (batch_depth[i]))
        # cv2.waitKey(1)

        return (label_map,
                label_map)
