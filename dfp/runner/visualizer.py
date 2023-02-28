from __future__ import print_function
from __future__ import absolute_import

from .base.runner import Runner
from ..img_utils.utils import color_map

from ..agents.multiagent import MultiAgent
from ..mapping.multimapper import MultiMapper

import numpy as np
import logging
import time
import cv2

import visdom


class Visualizer(Runner):
    """
    Visualize the decisions the current network plays
    Was called "Inference" in the original DFP
    """

    def __init__(self, experiment_cfg, future_steps=np.array([1, 2, 4, 8, 16, 32])):
        # NOTE: Hack the number of agents, in order to visualize the input/outputs
        experiment_cfg.num_agents = 1
        if (experiment_cfg.components.use_automap_net and
                (not experiment_cfg.components.use_doom_automap)):
            mapper = MultiMapper(experiment_cfg)
        else:
            mapper = None

        super(Visualizer, self).__init__(experiment_cfg, future_steps=future_steps)

        # NOTE: Constants
        self.vis_window = 300
        self.scale = 5

        self.mapper = mapper

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

        self.agent.init_simulators(experiment_cfg, visible=True)  # For visualization purposes

        # TODO: Better solution for visualizing map, if it's not going to be rendered
        if experiment_cfg.img_shape is not None:
            self.full_img_shape = ((1, experiment_cfg.img_shape[0]) +
                                   (experiment_cfg.img_shape[1] * self.scale,
                                    experiment_cfg.img_shape[1] * self.scale))
        else:
            self.full_img_shape = (1, 1, 84 * self.scale, 84 * self.scale)

        if experiment_cfg.map_shape is not None:
            self.full_map_shape = ((1, 1) +
                                   (experiment_cfg.map_shape[1] * self.scale,
                                    experiment_cfg.map_shape[2] * self.scale))
        else:
            self.full_map_shape = (1, 1, 42 * self.scale, 42 * self.scale)

        self.future_step = future_steps

        # Visdom instance
        self.vis = visdom.Visdom()
        self.vis.close()

        # visualizations
        self.img_win_opts = {'title': 'First Person Image'}
        self.depth_win_opts = {'title': 'Depth Image'}
        self.labels_win_opts = {'title': 'Label Image'}

        if self.agent.simulators[0].num_available_variables == 3:
            self.meas_win_opts = [{'title': 'AMMO'}, {'title': 'HEALTH'}, {'title': 'FRAGS'}]
        else:
            self.meas_win_opts = [{'title': 'HEALTH'}]

        self.automap_win_opts = {'title': 'Map'}
        self.automap_labels_win_opts = {'title': 'Map, Labels'}
        self.rwrd_win_opts = {'title': 'Rewards'}

        if self.use_img:
            self.img_win = self.vis.image(np.zeros(self.full_img_shape[1:]),
                                          opts=self.img_win_opts)
        # else:
        #     # NOTE: Should be able to use the image regardless of whether if it's an actual input or not
        #     # If not, maybe we should try rendering ithe front facing view...
        #     self.img_win_opts['title'] += ' (unused)'
        #     self.img_win = self.vis.image(np.zeros(experiment_cfg.img_shape),
        #                                   opts=self.img_win_opts)

        if self.use_depth:
            self.depth_win = self.vis.image(np.zeros(self.full_img_shape[1:]),
                                            opts=self.depth_win_opts)

        if self.use_labels:
            self.labels_win = self.vis.image(np.zeros(self.full_img_shape[1:]),
                                             opts=self.labels_win_opts)

        if self.use_automap:
            img = color_map(np.zeros(self.full_map_shape[1:])[0]).transpose(2, 0, 1)
            self.automap_win = self.vis.image(img,
                                              opts=self.automap_win_opts)

        if self.use_automap_labels:
            img = color_map(np.zeros(self.full_map_shape[1:])[0]).transpose(2, 0, 1)
            self.automap_labels_win = self.vis.image(img,
                                                     opts=self.automap_labels_win_opts)

        self.meas_win = []
        self.meas = []
        for i in range(len(self.meas_win_opts)):
            self.meas_win_opts[i].update({'markers': True})
            self.meas_win.append(self.vis.line(X=np.array([0]),
                                               Y=np.array([0]),
                                               opts=self.meas_win_opts[i]))
            self.meas.append([])

        self.rwrd_win = self.vis.line(X=np.array([0]), Y=np.array([0]), opts=self.rwrd_win_opts)

        self.meas_inc = 0
        self.rwrd_inc = 0

        self.rwrds = []

    def initial_state(self):
        unscaled_img_shape = (self.full_img_shape[0],
                              self.full_img_shape[1],
                              int(self.full_img_shape[2] / self.scale),
                              int(self.full_img_shape[3] / self.scale))

        unscaled_map_shape = (self.full_map_shape[0],
                              1,  # Will be one hot encoded
                              int(self.full_map_shape[2] / self.scale),
                              int(self.full_map_shape[3] / self.scale))

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
                state_automaps, state_automaps_labels, state_meas)

    def resize(self, img, input_map=False, input_img=False, id=None):
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

        if id is not None:
            cv2.imshow(id, color_map(img))

        if len(img.shape) == 2:
            img = img[:, :, None]

        img = img.transpose(2, 0, 1)

        return img

    def state_visualize(self, imgs, depths, labels, automaps, automap_labels, meas):
        # NOTE: Should be independent of image or not...
        if self.use_img:
            self.vis.image(self.resize(imgs[0], input_img=True),
                           win=self.img_win, opts=self.img_win_opts)

        if self.use_depth:
            self.vis.image(self.resize(depths[0], input_img=True),
                           win=self.depth_win, opts=self.depth_win_opts)

        if self.use_labels:
            self.vis.image(self.resize(labels[0], input_img=True),
                           win=self.labels_win, opts=self.labels_win_opts)

        if self.use_automap:
            automap = self.resize(automaps[0], input_map=True, id='automap')

            self.vis.image(automap,
                           win=self.automap_win, opts=self.automap_win_opts)

        if self.use_automap_labels:
            automap_labels = self.resize(automap_labels[0],
                                         input_map=True, id='automap_labels')
            self.vis.image(automap_labels,
                           win=self.automap_labels_win,
                           opts=self.automap_labels_win_opts)

        for i in range(len(self.meas_win)):
            if self.meas_inc >= self.vis_window:
                self.meas[i].pop(0)
                self.meas[i].append(meas[0][i])
                self.vis.line(X=np.arange(self.vis_window),
                              Y=np.asarray(self.meas[i]),
                              win=self.meas_win[i],
                              opts=self.meas_win_opts[i])
            else:
                self.meas[i].append(meas[0][i])
                self.vis.line(X=np.array([self.meas_inc]),
                              Y=np.array([meas[0][i]]),
                              win=self.meas_win[i],
                              opts=self.meas_win_opts[i],
                              update='append')

        self.meas_inc += 1
        cv2.waitKey(1)

    def rwrd_visualize(self, accum_rwrd):
        if self.rwrd_inc >= self.vis_window:
            self.rwrds.pop(0)
            self.rwrds.append(accum_rwrd)
            self.vis.line(X=np.arange(self.vis_window),
                          Y=np.asarray(self.rwrds),
                          win=self.rwrd_win,
                          opts=self.rwrd_win_opts)
        else:
            self.rwrds.append(accum_rwrd)
            self.vis.line(X=np.array([self.rwrd_inc]),
                          Y=np.array([accum_rwrd]),
                          win=self.rwrd_win,
                          opts=self.rwrd_win_opts,
                          update='append')
        self.rwrd_inc += 1

    def visualize_time(self, total_time=120):
        """
        Vizualize the current results that the agent has learned
        """
        start = time.time()
        while time.time() - start < total_time:
            self.episode()

    def visualize_episodes(self, total_episodes):
        all_rwrds = []
        for _ in xrange(total_episodes):
            rwrd = self.episode()
            all_rwrds.append(rwrd)

        all_rwrds = np.asarray(all_rwrds)
        print("Average across %d Episodes" % (total_episodes))
        print("Max Reward: %f" % (all_rwrds.max()))
        print("Min Reward: %f" % (all_rwrds.min()))
        print("Average Reward: %f" % (all_rwrds.mean()))
        print("Stdev Reward: %f" % (all_rwrds.std()))

        return all_rwrds

    def episode(self):
        imgs, depths, labels, automaps, automap_labels, meas = self.initial_state()
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
            self.state_visualize(*(new_state[:-2]))
            imgs, depths, labels, automaps, automap_labels, meas, rwds, terms = new_state
            accum_rwrd += rwds[0]

            if terms[0]:
                self.rwrd_visualize(accum_rwrd)
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
