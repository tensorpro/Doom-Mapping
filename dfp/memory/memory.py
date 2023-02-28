'''
Class for experience replay with multiple actors
'''

from __future__ import print_function
import numpy as np
import random
import logging

from ..targets.target import TargetMaker
from ..torch_utils import torch_functions


class Memory(object):
    def __init__(self, experiment_cfg, meas_length, num_actions,
                 future_steps=np.array([1, 2, 4, 8, 16, 32]),
                 min_meas_num=3, capacity=20000):
        # params
        self.capacity = capacity
        self.history_length = 1
        self.history_step = 1

        # TODO: Replace this with a more dynamic "target maker" type of thing
        if meas_length == 1:
            self.obj_shape = (len([0.5, 0.5, 1.0]),)
        else:
            self.obj_shape = (len([0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 1.]),)

        self.num_heads = experiment_cfg.num_agents
        self.target_maker = TargetMaker(meas_length, future_steps,
                                        min_meas_num=min_meas_num)

        self.head_offset = int(self.capacity / self.num_heads)

        self.img_shape = experiment_cfg.img_shape
        self.map_shape = experiment_cfg.map_shape

        self.meas_shape = (meas_length,)
        self.action_shape = (num_actions,)
        # Each image will have one "channel", unless rgb is used
        if self.img_shape is not None:
            self.state_imgs_shape = (self.history_length * 1,) + self.img_shape[1:]
            logging.info("Experience Buffer Image Shape: %s" % (str(self.img_shape)))

        self.state_meas_shape = (self.history_length * self.meas_shape[0],)

        if self.map_shape is not None:
            self.state_automaps_shape = (self.history_length * 1,) + self.map_shape[1:]
            logging.info("Experience Buffer Map Shape: %s" % (str(self.map_shape)))

        # Keep track of what to save
        COMPONENTS = ''
        self.save_img = experiment_cfg.components.save_img
        if self.save_img:
            COMPONENTS += ' IMAGE'

        self.save_depth = experiment_cfg.components.save_depth
        if self.save_depth:
            COMPONENTS += ' DEPTH'

        self.save_labels = experiment_cfg.components.save_labels
        if self.save_labels:
            COMPONENTS += ' LABELS'

        self.save_automap = experiment_cfg.components.save_automap
        if self.save_automap:
            COMPONENTS += ' AUTOMAP'

        self.save_automap_labels = experiment_cfg.components.save_automap_labels
        if self.save_automap_labels:
            COMPONENTS += ' AUTOMAP_LABELS'

        logging.info('Experience Replay Buffer is saving:%s' % (COMPONENTS))

        # initialize dataset
        self.reset()

    def reset(self):
        if self.save_img:
            self._images = np.zeros(
                shape=(self.capacity,) + (1, self.img_shape[1], self.img_shape[2]), dtype=np.uint8)

        if self.save_depth:
            self._depths = np.zeros(
                shape=(self.capacity,) + (1, self.img_shape[1], self.img_shape[2]), dtype=np.uint8
            )

        if self.save_labels:
            self._labels = np.zeros(
                shape=(self.capacity,) + (1, self.img_shape[1], self.img_shape[2]), dtype=np.uint8
            )

        self._measurements = np.zeros(
            shape=(self.capacity,) + self.meas_shape, dtype=np.float32)
        self._rewards = np.zeros(
            shape=(self.capacity,), dtype=np.float32)
        self._terminals = np.ones(
            shape=(self.capacity,), dtype=np.bool)
        self._actions = np.zeros(
            shape=(self.capacity,) + self.action_shape, dtype=np.int)
        self._objectives = np.zeros(
            shape=(self.capacity,) + self.obj_shape, dtype=np.float32)
        # this is needed to compute future targets efficiently
        self._n_episode = np.zeros(
            shape=(self.capacity,), dtype=np.uint64)
        # this is needed to compute future targets efficiently
        self._n_head = np.zeros(
            shape=(self.capacity,), dtype=np.uint64)

        self._curr_indices = np.arange(self.num_heads) * int(self.head_offset)
        self._episode_counts = np.zeros(self.num_heads)

        if self.save_automap:
            self._automaps = np.zeros(
                shape=(self.capacity, 1) + self.map_shape[1:], dtype=np.uint8
            )

        if self.save_automap_labels:
            self._automaps_labels = np.zeros(
                shape=(self.capacity, 1) + self.map_shape[1:], dtype=np.uint8
            )

    def add_batch(self, imgs, depths, labels, automaps, automaps_labels, meass, rwrds, terms, acts, objs=None, preds=None):
        ''' Add experience to dataset.

        Args:
            img: single observation frame
            meas: extra measurements from the state
            rwrd: reward
            term: terminal state
            act: action taken
        '''

        acts = np.array([torch_functions.to_sparse(i, self.action_shape[0]) for i in acts])

        if self.save_img:
            self._images[self._curr_indices] = imgs
        if self.save_depth:
            self._depths[self._curr_indices] = depths
        if self.save_labels:
            self._labels[self._curr_indices] = labels
        self._measurements[self._curr_indices] = meass
        self._rewards[self._curr_indices] = rwrds
        self._terminals[self._curr_indices] = terms
        self._actions[self._curr_indices] = np.array(acts)
        if isinstance(objs, np.ndarray):
            self._objectives[self._curr_indices] = objs
        if isinstance(preds, np.ndarray):
            self._predictions[self._curr_indices] = preds
        if self.save_automap:
            self._automaps[self._curr_indices] = automaps
        if self.save_automap_labels:
            self._automaps_labels[self._curr_indices] = automaps_labels

        self._n_episode[self._curr_indices] = self._episode_counts
        # this is a hack to simulate our version of ViZDoom which gives the agent
        # the first post-mortem measurement. This turns out to matter a bit for
        # learning
        terminated = np.where(np.array(terms) == True)[0]
        term_inds = self._curr_indices[terminated]
        self._measurements[term_inds] = self._measurements[(term_inds - 1) % self.capacity]
        if self.meas_shape[0] == 1:
            for ti in term_inds:
                if self._measurements[ti, 0] < 8.:
                    self._measurements[ti, 0] -= 8.
        if self.meas_shape[0] == 3:
            for ti in term_inds:
                if self._measurements[ti, 1] < 12.:
                    self._measurements[ti, 1] -= 12.
        # in case there are 2 terminals in a row - not sure this actually can
        # happen, but just in case
        prev_terminals = np.where(self._terminals[(term_inds - 1) % self.capacity])[0]
        if len(prev_terminals) > 0:
            print('Prev terminals', prev_terminals)
        self._measurements[term_inds[prev_terminals]] = 0.
        # end of hack
        # self._n_episode[term_inds] = self._episode_counts[terminated]+100# so
        # that the terminal step of the episode is not used as a target by
        # target_maker
        self._n_head[self._curr_indices] = np.arange(self.num_heads)

        self._episode_counts = self._episode_counts + (np.array(terms) == True)
        self._curr_indices = (self._curr_indices + 1) % self.capacity
        # make the following state terminal, so that our current episode doesn't
        # get stitched with the next one when sampling states
        self._terminals[self._curr_indices] = True

    def get_states(self, indices):
        frames = np.zeros(len(indices) * self.history_length, dtype=np.int64)
        for (ni, index) in enumerate(indices):
            frame_slice = np.arange(int(index) - self.history_length * self.history_step +
                                    1, (int(index) + 1), self.history_step) % self.capacity
            frames[ni * self.history_length:(ni + 1) * self.history_length] = frame_slice

        state_imgs = None
        state_depths = None
        state_labels = None

        if self.save_img:
            state_imgs = np.reshape(np.take(self._images, frames, axis=0), (len(
                indices),) + self.state_imgs_shape)
        if self.save_depth:
            state_depths = np.reshape(np.take(self._depths, frames, axis=0), (len(
                indices),) + self.state_imgs_shape)
        if self.save_labels:
            state_labels = np.reshape(np.take(self._labels, frames, axis=0), (len(
                indices),) + self.state_imgs_shape)

        state_meas = np.reshape(np.take(self._measurements, frames, axis=0),
                                (len(indices),) + self.state_meas_shape)

        state_automaps = None
        if self.save_automap:
            state_automaps = np.reshape(np.take(self._automaps, frames, axis=0), (len(
                indices),) + self.state_automaps_shape)

        state_automaps_labels = None
        if self.save_automap_labels:
            state_automaps_labels = np.reshape(np.take(self._automaps_labels, frames, axis=0), (len(
                indices),) + self.state_automaps_shape)

        return (state_imgs, state_depths, state_labels,
                state_automaps, state_automaps_labels, state_meas)

    def get_current_state(self):
        '''  Return most recent observation sequence '''
        return self.get_states(list((self._curr_indices - 1) % self.capacity))

    def get_last_indices(self):
        '''  Return most recent indices '''
        return list((self._curr_indices - 1) % self.capacity)

    def get_targets(self, indices):
        # TODO this 12345678 is a hack, but should be good enough
        return self.target_maker.make_targets(indices, self._measurements, self._rewards, self._n_episode + 12345678 * self._n_head)

    def has_valid_history(self, index):
        return (not self._terminals[np.arange(int(index) - self.history_length * self.history_step + 1, int(index) + 1) % self.capacity].any())

    def curr_states_with_valid_history(self):
        return [self.has_valid_history((ind - 1) % self.capacity) for ind in list(self._curr_indices)]

    def has_valid_target(self, index):
        return (not self._terminals[np.arange(index, index + self.target_maker.min_future_frames + 1) % self.capacity].any())

    def is_valid_state(self, index):
        return self.has_valid_history(index) and self.has_valid_target(index)

    def get_observations(self, indices):
        indices_arr = np.array(indices)
        state_imgs, state_depths, state_labels, state_automaps, state_automaps_labels, state_meas = self.get_states(
            (indices_arr - 1) % self.capacity)
        rwrds = self._rewards[indices_arr]
        acts = self._actions[indices_arr]
        terms = self._terminals[indices_arr].astype(int)
        targs = self.get_targets((indices_arr - 1) % self.capacity)
        if isinstance(self._objectives, np.ndarray):
            objs = self._objectives[indices_arr]
        else:
            objs = None

        return (state_imgs, state_depths, state_labels,
                state_automaps, state_automaps_labels, state_meas, rwrds, terms, acts, targs, objs)

    def get_random_batch(self, batch_size):
        ''' Sample minibatch of experiences for training '''

        samples = []  # indices of the end of each sample

        while len(samples) < batch_size:
            index = random.randrange(self.capacity)
            # check if there is enough history to make a state and enough future to make targets
            if self.is_valid_state(index):
                samples.append(index)
            else:
                continue

        # create batch
        return self.get_observations(np.array(samples))

    def compute_avg_meas_and_rwrd(self, start_idx, end_idx):
        # compute average measurement values per episode, and average cumulative reward per episode
        curr_num_obs = 0.
        curr_sum_meas = self._measurements[0] * 0
        curr_sum_rwrd = self._rewards[0] * 0
        num_episodes = 0.
        total_sum_meas = self._measurements[0] * 0
        total_sum_rwrd = self._rewards[0] * 0
        for index in range(int(start_idx), int(end_idx)):
            curr_sum_rwrd += self._rewards[index]
            if self._terminals[index]:
                if curr_num_obs:
                    total_sum_meas += curr_sum_meas / curr_num_obs
                    total_sum_rwrd += curr_sum_rwrd
                    num_episodes += 1
                curr_sum_meas = self._measurements[0] * 0
                curr_sum_rwrd = self._rewards[0] * 0
                curr_num_obs = 0.
            else:
                curr_sum_meas += self._measurements[index]
                curr_num_obs += 1

        if num_episodes == 0.:
            total_avg_meas = curr_sum_meas / curr_num_obs
            total_avg_rwrd = curr_sum_rwrd
        else:
            total_avg_meas = total_sum_meas / num_episodes
            total_avg_rwrd = total_sum_rwrd / num_episodes

        return total_avg_meas, total_avg_rwrd
