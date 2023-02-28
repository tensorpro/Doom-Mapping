from __future__ import print_function
from __future__ import absolute_import

from ..memory.memory import Memory
from .evaluator import Evaluator
from ..simulator.DoomSimulator import DoomSimulator
from ..torch_utils import torch_functions
from ..mapping import multimapper


import numpy as np
import logging

import torch
import time

import gc


class Trainer(Evaluator):
    def __init__(self, experiment_cfg, mapper=None,
                 train_capacity=20000, test_capacity=55000,
                 future_steps=np.array([1, 2, 4, 8, 16, 32]),
                 min_meas_num=3, num_steps=820000,
                 batch_size=64):

        # Temporary Simulator to parse config file
        simulator = DoomSimulator(experiment_cfg)

        if (mapper is None and
                experiment_cfg.components.use_automap_net and
                (not experiment_cfg.components.use_doom_automap)):
            mapper = multimapper.MultiMapper(experiment_cfg)
        else:
            mapper = None

        memory = Memory(experiment_cfg, simulator.num_available_variables,
                        simulator.num_available_buttons, future_steps,
                        min_meas_num, capacity=train_capacity)

        super(Trainer, self).__init__(experiment_cfg,
                                      test_capacity=test_capacity, future_steps=future_steps,
                                      batch_size=batch_size, mapper=mapper)

        self.train_memory = memory

        # Swap memory to training one
        self.swap_memory(self.train_memory)

        # Learning
        self.learning_rate = 0.0001
        self.decay_steps = 250000
        self.decay_rate = 0.3

        # Custom implementation, taken from Tensorflow
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          betas=(0.95, 0.999), lr=self.learning_rate, eps=1e-4)
        self.num_steps = num_steps

    def learn(self):
        """
        Learning phase. Learn from the current memory provided.
        """
        self.multiagent.fill(self.steps_taken, inference=False)
        logging.info('Training network')
        # Now train using the network
        loss = 0
        steps_start = self.steps_taken
        for i in xrange(self.num_steps):
            if i % 7812 == 0:
                logging.info("Testing Policy at Step: %d" %
                             (self.multiagent.batch_size * self.steps_taken))
                self.test()

            curr_lr = torch_functions.adjust_learning_rate(
                self.optimizer, self.learning_rate,
                self.decay_rate, self.decay_steps, self.steps_taken
            )

            loss += self.multiagent.train_one_batch(self.optimizer)

            curr_eps = torch_functions.calculate_eps(self.steps_taken)

            self.multiagent.n_step(8, self.steps_taken, curr_eps, write_logs=False)
            self.steps_taken += 1

            if i % 651 == 0:
                logging.info("Finished up to training Policy at Step: %d, loss: %5.5f." %
                             (self.multiagent.batch_size * self.steps_taken, loss / (self.steps_taken - steps_start)))
                logging.info("Current Learning rate: %f" % (curr_lr))
                curr_eps = torch_functions.calculate_eps(self.steps_taken)
                logging.info("Current epsilon: %5.5f" % (curr_eps))
                steps_start = self.steps_taken
                loss = 0
                self.save()

                gc.collect()

        logging.info("Testing Final Policy at Step: %d" %
                     (self.multiagent.batch_size * self.steps_taken))
        self.test()
        self.save()

    def run(self):
        self.learn()
