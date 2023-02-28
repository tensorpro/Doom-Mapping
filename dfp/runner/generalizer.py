from __future__ import print_function
from __future__ import absolute_import

from .base.runner import Runner

from ..agents.multiagent import MultiAgent
from ..mapping.multimapper import MultiMapper
from ..mapping.vizdoom_pointcloud import VDMapper

import numpy as np
import logging
import time
import cv2


class Generalizer(Runner):
    """
    Run phases of evaluation, without the image outputs
    """

    def __init__(self, experiment_cfg, future_steps=np.array([1, 2, 4, 8, 16, 32]),
                 produce_map=False):
        # NOTE: Hack the number of agents, in order to visualize the input/outputs
        experiment_cfg.num_agents = 1
        if (experiment_cfg.components.use_automap_net and
                (not experiment_cfg.components.use_doom_automap)):
            mapper = MultiMapper(experiment_cfg)
        else:
            mapper = None

        super(Generalizer, self).__init__(experiment_cfg, future_steps=future_steps)

        self.mapper = mapper

        if produce_map:
            experiment_cfg.components.use_gt_motion = True
            experiment_cfg.components.use_gt_depth = True

        self.agent = MultiAgent(experiment_cfg, self.net, None, mapper=self.mapper)
        # NOTE: A couple of custom configuration items for the game. Note that this will change the behvaior
        # TODO: ^

        COMPONENTS = ''
        self.use_img = experiment_cfg.components.save_img
        if self.use_img:
            COMPONENTS += ' IMAGE'

        self.use_depth = experiment_cfg.components.save_depth
        if self.use_depth:
            COMPONENTS += ' DEPTH'

        self.use_labels = experiment_cfg.components.save_labels
        if self.use_labels:
            COMPONENTS += ' LABELS'

        self.use_automap = experiment_cfg.components.save_automap
        if self.use_automap:
            COMPONENTS += ' AUTOMAP'

        self.use_automap_labels = experiment_cfg.components.save_automap_labels
        if self.use_automap_labels:
            COMPONENTS += ' AUTOMAP_LABELS'

        logging.info('Visualizer Uses at input:%s' % (COMPONENTS))

        self.agent.init_simulators(experiment_cfg, visible=False)

        # TODO: Better solution for visualizing map, if it's not going to be rendered
        if experiment_cfg.img_shape is not None:
            self.full_img_shape = ((1, 1) +
                                   (experiment_cfg.img_shape[1],
                                    experiment_cfg.img_shape[2]))
        else:
            self.full_img_shape = (1, 1, 84, 84)

        if experiment_cfg.map_shape is not None:
            self.full_map_shape = ((1, 1) +
                                   (experiment_cfg.map_shape[1],
                                    experiment_cfg.map_shape[2]))
        else:
            self.full_map_shape = (1, 1, 42, 42)

        self.futures_step = future_steps

        # NOTE: Generate heat maps
        if produce_map:
            self.heat_map = VDMapper(experiment_cfg)
        else:
            self.heat_map = None

    def initial_state(self):
        unscaled_img_shape = (self.full_img_shape[0],
                              self.full_img_shape[1],
                              int(self.full_img_shape[2]),
                              int(self.full_img_shape[3]))

        unscaled_map_shape = (self.full_map_shape[0],
                              1,  # Will be one hot encoded
                              int(self.full_map_shape[2]),
                              int(self.full_map_shape[3]))

        if self.use_img:
            state_imgs = np.zeros(unscaled_img_shape, dtype=np.uint8)
        else:
            state_imgs = None

        if self.use_depth:
            state_depths = np.zeros(unscaled_img_shape, dtype=np.uint8)
        else:
            state_depths = None

        if self.use_labels:
            state_labels = np.zeros(unscaled_img_shape, dtype=np.uint8)
        else:
            state_labels = None

        state_meas = np.zeros(self.agent.simulators[0].num_available_variables)

        if self.use_automap:
            state_automaps = np.zeros(unscaled_map_shape, dtype=np.uint8)
        else:
            state_automaps = None

        if self.use_automap_labels:
            state_automaps_labels = np.zeros(unscaled_map_shape, dtype=np.uint8)
        else:
            state_automaps_labels = None

        return (state_imgs, state_depths, state_labels,
                state_automaps, state_automaps_labels, state_meas,
                np.zeros(4))

    def resize(self, img, input_map=False, input_img=False):
        if not (input_map ^ input_img):  # Only one of them should be True
            raise ValueError, "Resize must be done on either maps or images"

        img = np.asarray(img, dtype=np.uint8)

        if input_map:
            # img = color_map(img[0])
            img = img[0]
            img = cv2.resize(img,
                             self.full_map_shape[:1:-1],
                             interpolation=cv2.INTER_NEAREST)

        elif input_img:
            img = cv2.resize(img.transpose(1, 2, 0),
                             self.full_img_shape[:1:-1],
                             interpolation=cv2.INTER_NEAREST)

        if len(img.shape) == 2:
            img = img[:, :, None]

        img = img.transpose(2, 0, 1)

        return img

    def visualize_time(self, total_time=120):
        """
        Vizualize the current results that the agent has learned
        """
        start = time.time()
        while time.time() - start < total_time:
            self.episode()

    def visualize_episodes(self, total_episodes):
        all_rwrds = []
        for ep in xrange(total_episodes):
            print("episode: %d" % (ep))
            rwrd = self.episode()
            all_rwrds.append(rwrd)

        all_rwrds = np.asarray(all_rwrds)
        print("Average across %d Episodes" % (total_episodes))
        print("Max Reward: %f" % (all_rwrds.max()))
        print("Min Reward: %f" % (all_rwrds.min()))
        print("Average Reward: %f" % (all_rwrds.mean()))
        print("Stdev Reward: %f" % (all_rwrds.std()))

        if self.heat_map:
            return all_rwrds, self.heat_map.global_map.current_map
        else:
            return all_rwrds

    def episode(self):
        imgs, depths, labels, automaps, automap_labels, meas, coords = self.initial_state()
        accum_rwrd = 0
        while True:
            # NOTE: Typically memory handles this filtering elegantly, but visualization doesn't have that
            if not self.use_automap:
                automaps = None

            if not self.use_automap_labels:
                automap_labels = None

            curr_act, _ = self.net.apply_inference(imgs, depths,
                                                   labels, meas,
                                                   automap=automaps,
                                                   automap_labels=automap_labels,
                                                   preprocess=True, CUDA=self.CUDA)

            new_state = self.agent.step_all(curr_act)
            imgs, depths, labels, automaps, automap_labels, meas, rwds, terms, coords = new_state
            # print(depths.shape)
            # print(coords)
            if self.heat_map:
                # Sometimes the depth buffer isn't extracted properly.
                state = self.agent.simulators[0].game.get_state()
                self.heat_map.compute_global_map_prob(state.depth_buffer,
                                                      coords[0],
                                                      terms[0])

            accum_rwrd += rwds[0]

            if terms[0]:
                return accum_rwrd

    def run(self, episodes=-1):
        logging.info("Visualizing at step: %d" % (self.steps_taken))
        self.net.eval()

        if episodes <= 0:
            logging.info('Visualizing for Time')
            rwrd = self.visualize_time()
        else:
            rwrd = self.visualize_episodes(episodes)

        return rwrd
