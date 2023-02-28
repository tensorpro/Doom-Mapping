from __future__ import print_function
from __future__ import absolute_import

from ..simulator.DoomSimulator import DoomSimulator
from ..torch_utils import torch_functions
from ..img_utils.utils import rgb2gray_batch
from ..memory.preprocessors import Preprocessors
from ..networks.dense import resnet_load
from ..cfg import CFG, VARIABLES

import torch
from torch.autograd import Variable
import numpy as np
import cv2

import time
import logging

import os


class MultiAgent(object):
    """
    Multi-agent manages the actual training loop, as well as joining and computing losses.
    It also creates threads and pools them together.
    """

    def __init__(self, experiment_cfg, net, memory, batch_size=64, mapper=None):
        # Network
        self.net = net

        # Training related
        self.num_agents = experiment_cfg.num_agents

        self.batch_size = batch_size
        self.CUDA = experiment_cfg.use_gpu

        # Auxiliary variables
        self.save_dir = experiment_cfg.save_dir
        self.name = experiment_cfg.name

        self.use_pred_nets = experiment_cfg.components.use_pred_nets

        self.use_pred_depth = experiment_cfg.components.use_pred_depth
        self.use_pred_labels = experiment_cfg.components.use_pred_labels
        self.use_pred_motion = experiment_cfg.components.use_pred_motion

        # Prediction Networks
        if self.use_pred_depth:
            # TODO: Load the right depth prediction network
            self.depth_net = None
            raise NotImplementedError, "Depth Prediction not implemented"
        else:
            self.depth_net = None

        if self.use_pred_labels:
            # TODO: Load the right label prediction network
            self.label_net = None
            raise NotImplementedError, "Label Prediction not implemented"
        else:
            self.label_net = None

        if self.use_pred_motion:
            # TODO: Load the right motion prediction network
            self.motion_net = None
            raise NotImplementedError, "Motion Prediction not implemented"
        else:
            self.motion_net = None

        # Mapping manager
        self.mapper = mapper

        # Keep track of memory
        self.memory = memory

        # Configuration
        # self.components = experiment_cfg.components

    def init_simulators(self, simulator_cfg, visible=False):
        assert not hasattr(self, 'simulators'), "Simulators already exist"
        self.simulators = [DoomSimulator(simulator_cfg, visible=visible)
                           for i in range(self.num_agents)]
        for sim in self.simulators:
            sim.init()

    def swap_memory(self, memory):
        self.memory = memory

    def reset_simulator(self):
        for simulator in self.simulators:
            simulator.reset()

    def reset(self):
        """Reset the current memory"""
        print('Resetting multiagent')
        self.memory.reset()
        if self.mapper is not None:
            self.mapper.reset()

    def fill(self, steps_taken, inference=True):
        """
        Fill the replay buffer with experience.
        """
        logging.info("Filling up the replay buffer hold on...")
        start = time.time()
        if inference:
            self.n_step(self.memory.capacity / self.num_agents,
                        steps_taken * self.batch_size,
                        0.0,
                        write_logs=True)
        else:
            curr_eps = torch_functions.calculate_eps(steps_taken)
            self.n_step(self.memory.capacity / self.num_agents,
                        steps_taken * self.batch_size,
                        curr_eps,
                        write_logs=False)

        logging.info("Finished in %5.5f seconds" % (time.time() - start))

    def step_all(self, index):
        RET = []
        PRED_NET = []
        MAPPING = []

        states = [sim.make_action(index[i]) for i, sim in enumerate(self.simulators)]
        raw_gt_imgs, raw_gt_depths, raw_gt_labels, measurements, gt_coords, rwds, terms, automap_img = zip(
            *states)
        raw_gt_imgs = np.asarray(raw_gt_imgs, dtype=np.uint8)

        # NOTE: Check to see if values have items, and convert them to a numpy array
        if raw_gt_depths is not None and raw_gt_depths[0] is not None:
            raw_gt_depths = np.asarray(raw_gt_depths, dtype=np.uint8)

        if raw_gt_labels is not None and raw_gt_labels[0] is not None:
            raw_gt_labels = np.asarray(raw_gt_labels, dtype=np.uint8)

        gt_coords = np.asarray(gt_coords)

        measurements = np.asarray(measurements, dtype=np.float32)
        rwds = np.asarray(rwds, dtype=np.float32)
        terms = np.asarray(terms, dtype=np.bool)

        # NOTE: Network First-Person Image
        if self.net.use_img:
            if self.use_pred_nets:
                # convert to grey scale and resize
                gray_imgs = np.expand_dims(rgb2gray_batch(raw_gt_imgs.transpose(0, 2, 3, 1)), 1)
            else:
                gray_imgs = raw_gt_imgs

            gray_imgs = np.asarray([cv2.resize(img[0], self.simulators[0].img_resolution[::-1])[None, :, :]
                                    for img in raw_gt_imgs], dtype=np.uint8)
            RET.append(gray_imgs)
        else:
            RET.append(None)

        if self.use_pred_depth or self.use_pred_labels:
            PRED_NET.append(raw_gt_imgs)

        # TODO: Calculate Dependencies for Motion
        if self.use_pred_motion:
            pass

        # NOTE: Calculate Predictions
        if self.use_pred_nets:
            # color image is returned
            # scale to 84x84
            np_imgs = raw_gt_imgs.transpose((0, 3, 1, 2))
            inputs = Variable(torch.from_numpy(np_imgs / 255.).float(), volatile=True)

            if self.CUDA:
                inputs = inputs.cuda()

            # Depending on what parameters gets passed in, use those
            if self.use_pred_depth:
                pred_depths = self.depth_net(inputs).data.cpu().numpy()
            else:
                pred_depths = None

            if self.use_pred_labels:
                pred_labels = self.label_net(inputs).data.cpu().numpy()
            else:
                pred_labels = None

        # NOTE: Calculate Network Dependencies for depth, labels
        if self.net.use_gt_depth:
            depths = np.asarray([cv2.resize(img[0], self.simulators[0].img_resolution[::-1])[None, :, :]
                                 for img in raw_gt_depths], dtype=np.uint8)
            RET.append(depths)
        elif self.net.use_pred_depth:
            depths = np.asarray([cv2.resize(img[0], self.simulators[0].img_resolution[::-1])[None, :, :]
                                 for img in pred_depths], dtype=np.uint8)
            RET.append(depths)
        else:
            RET.append(None)

        if self.net.use_gt_labels:
            labels = np.asarray([cv2.resize(img[0], self.simulators[0].img_resolution[::-1])[None, :, :]
                                 for img in raw_gt_labels], dtype=np.uint8)
            RET.append(labels)
        elif self.net.use_pred_labels:
            labels = np.asarray([cv2.resize(img[0], self.simulators[0].img_resolution[::-1])[None, :, :]
                                 for img in pred_labels], dtype=np.uint8)
            RET.append(labels)
        else:
            RET.append(None)

        # NOTE: Calculate Mapping Dependencies
        if self.mapper is not None:  # None if we aren't using it
            if self.mapper.use_gt_depth:
                MAPPING.append(raw_gt_depths)
            elif self.mapper.use_pred_depth:
                MAPPING.append(pred_depths)
            else:
                MAPPING.append(None)

            if self.mapper.use_gt_labels:
                MAPPING.append(raw_gt_labels)
            elif self.mapper.use_pred_labels:
                MAPPING.append(pred_labels)
            else:
                MAPPING.append(None)

            if self.mapper.use_gt_motion:
                MAPPING.append(gt_coords)
            elif self.mapper.use_pred_motion:
                # TODO: Actually calculate pred coords
                # MAPPING.append(pred_coords)
                logging.warn("Predicted Motion is not implemented")
                MAPPING.append(None)
            else:
                MAPPING.append(None)

            MAPPING.append(terms)

        # NOTE: Mapping Calculation and Network Dependencies
        if self.mapper is not None:
            automaps, automap_labels = self.mapper(*MAPPING)
            RET.append(automaps)
            RET.append(automap_labels)
        elif self.simulators[0].use_doom_automap:
            # TODO: Check if this is correct
            automaps = np.asarray([cv2.resize(img[0], self.simulators[0].map_resolution[::-1])[None, :, :]
                                   for img in automap_img], dtype=np.uint8)
            RET.append(automaps)
            RET.append(None)
        else:
            automaps = None
            automap_labels = None
            RET.append(None)
            RET.append(None)

        # NOTE: Last bit of dependencies...
        RET.append(measurements)
        RET.append(rwds)
        RET.append(terms)
        RET.append(gt_coords)

        return RET

    def n_step(self, num_steps, global_step, curr_eps=0.0, write_logs=False):
        ns = 0
        last_meas = np.zeros((self.num_agents,) + self.memory.meas_shape)

        if write_logs:
            log_dir = os.path.join(self.save_dir, self.name)

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_prefix = os.path.join(log_dir, "log")

            log_brief = open(log_prefix + '_brief.txt', 'a')
            log_detailed = open(log_prefix + '_detailed.txt', 'a')

            log_detailed.write('Step {0}\n'.format(global_step))
            start_times = time.time() * np.ones(self.num_agents)
            num_episode_steps = np.zeros(self.num_agents)
            accum_rewards = np.zeros(self.num_agents)
            accum_meas = np.zeros((self.num_agents,) + self.memory.meas_shape)
            total_final_meas = np.zeros(self.memory.meas_shape)
            total_avg_meas = np.zeros(self.memory.meas_shape)
            total_accum_reward = 0
            total_start_time = time.time()
            num_episodes = 0
            meas_dim = np.prod(self.memory.meas_shape)
            log_brief_format = (' '.join([('{' + str(n) + '}') for n in range(5)]) + ' | ' +
                                ' '.join([('{' + str(n + 5) + '}') for n in range(meas_dim)]) + ' | ' +
                                ' '.join([('{' + str(n + 5 + meas_dim) + '}') for n in range(meas_dim)]) + '\n')
            log_detailed_format = (' '.join([('{' + str(n) + '}') for n in range(4)]) + ' | ' +
                                   ' '.join([('{' + str(n + 4) + '}') for n in range(meas_dim)]) + ' | ' +
                                   ' '.join([('{' + str(n + 4 + meas_dim) + '}') for n in range(meas_dim)]) + '\n')

        self.net.eval()
        if curr_eps < 0:
            curr_eps = 0.0

        for ns in xrange(num_steps):
            randomize = np.random.rand(1)[0] < curr_eps

            if randomize:
                curr_act = np.random.randint(0,
                                             self.simulators[0].num_available_buttons,
                                             size=self.num_agents)
            else:
                imgs, depths, labels, automaps, automap_labels, meas = self.memory.get_current_state()
                curr_act, _ = self.net.apply_inference(imgs, depths, labels, meas,
                                                       automap=automaps, automap_labels=automap_labels,
                                                       preprocess=True, CUDA=self.CUDA)

            invalid_states = np.logical_not(np.array(self.memory.curr_states_with_valid_history()))
            curr_act[invalid_states] = np.random.randint(0,
                                                         self.simulators[0].num_available_buttons,
                                                         size=np.sum(invalid_states))

            imgs, depths, labels, automaps, automap_labels, measurements, rwds, terms, coords = self.step_all(
                curr_act)
            self.memory.add_batch(imgs, depths, labels,
                                  automaps, automap_labels,
                                  measurements, rwds, terms, curr_act)

            if write_logs:
                last_indices = np.array(self.memory.get_last_indices())
                last_rewards = self.memory._rewards[last_indices]
                prev_meas = last_meas
                last_meas = self.memory._measurements[last_indices]
                last_terminals = self.memory._terminals[last_indices]
                last_meas[np.where(last_terminals)[0]] = 0
                accum_rewards += last_rewards
                accum_meas += last_meas
                num_episode_steps = num_episode_steps + 1
                terminated_simulators = list(np.where(last_terminals)[0])
                for ns in terminated_simulators:
                    num_episodes += 1
                    episode_time = time.time() - start_times[ns]
                    avg_meas = accum_meas[ns] / float(num_episode_steps[ns])
                    total_avg_meas += avg_meas
                    total_final_meas += prev_meas[ns]
                    total_accum_reward += accum_rewards[ns]
                    start_times[ns] = time.time()
                    log_detailed.write(log_detailed_format.format(
                        *([num_episodes, num_episode_steps[ns], episode_time, accum_rewards[ns]] + list(prev_meas[ns]) + list(avg_meas))))
                    accum_meas[ns] = 0
                    accum_rewards[ns] = 0
                    num_episode_steps[ns] = 0
                    start_times[ns] = time.time()
        if write_logs:
            if num_episodes == 0:
                num_episodes = 1
            log_brief.write(log_brief_format.format(*([global_step, time.time(), time.time() - total_start_time, num_episodes, total_accum_reward / float(num_episodes)] +
                                                      list(total_final_meas / float(num_episodes)) + list(total_avg_meas / float(num_episodes)))))
            log_brief.close()
            log_detailed.close()
        # print("single step: %f" % (time.time() - s))
        self.net.train()

    def train_one_batch(self, optimizer):
        # s = time.time()
        imgs, depths, labels, automaps, automap_labels, meas, rewards, terms, actions, targs, objs = self.memory.get_random_batch(
            self.batch_size)
        # print("Generting one batch %f" % (time.time() - s))
        # Preprocess the inputs
        imgs = Preprocessors.preprocess_images(imgs, depths, labels)
        meas = Preprocessors.preprocess_measurements(meas)
        targs = Preprocessors.preprocess_targets(targs)
        actions = Preprocessors.preprocess_actions(actions)
        automaps = Preprocessors.preprocess_automaps(automaps, automap_labels)

        optimizer.zero_grad()

        if imgs is not None:
            imgs = Variable(torch.from_numpy(imgs)).float()
            if self.CUDA:
                imgs = imgs.cuda()

        if automaps is not None:
            automaps = Variable(torch.from_numpy(automaps)).float()
            if self.CUDA:
                automaps = automaps.cuda()

        meas = Variable(torch.from_numpy(meas)).float()
        targs = Variable(torch.from_numpy(targs)).float()

        if self.CUDA:
            meas = meas.cuda()
            targs = targs.cuda()

        preds = self.net(imgs, meas, automaps)

        loss = torch_functions.loss(preds, actions, targs,
                                    CUDA=self.CUDA)

        loss.backward()
        optimizer.step()

        return loss.data.cpu()[0]
