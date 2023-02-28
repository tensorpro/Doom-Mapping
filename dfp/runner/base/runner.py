from __future__ import print_function
from __future__ import absolute_import

from ...networks.network import DFP
from ...simulator.DoomSimulator import DoomSimulator

import os
import logging

import numpy as np
import random

import torch


class Runner(object):
    """
    Defines a base class to run specific tasks.
    See Trainer.py, Evaluator.py, or Visualize.py for more info
    """

    def __init__(self, experiment_cfg,
                 future_steps=np.array([1, 2, 4, 8, 16, 32])):

        # Temporary Simulator to parse config file
        simulator = DoomSimulator(experiment_cfg)

        # Network set up
        self.img_resolution = experiment_cfg.img_shape
        self.map_resolution = experiment_cfg.map_shape

        self.net = DFP(experiment_cfg,
                       simulator.num_available_variables,
                       simulator.num_available_buttons,
                       len(future_steps))

        if experiment_cfg.use_gpu:
            self.net.cuda()

        self.CUDA = experiment_cfg.use_gpu
        self.steps_taken = 0

        self.save_dir = experiment_cfg.save_dir
        self.name = experiment_cfg.name

        try:
            self.load(cuda_to_cpu=(not experiment_cfg.use_gpu))
        except IOError:
            logging.warn(
                'Failed to load model, this may be expected behavior if the training just started')

    def load(self, cuda_to_cpu=False):
        path = os.path.join(self.save_dir, self.name, 'model.pth')
        logging.info('attempting to load from %s' % (path))
        if cuda_to_cpu:
            dict = torch.load(path, map_location=lambda store, loc: store)
        else:
            dict = torch.load(path)

        if hasattr(self, "optimizer"):
            # Set optimizer
            logging.info("Loading optimzier")
            self.optimizer.load_state_dict(dict['optimizer'])
            self.learning_rate = dict['lr']
            # Set random seed
            np.random.set_state(dict['np_random_state'])
            random.setstate(dict['py_random_state'])
            torch.set_rng_state(dict['torch_random_state'])

        self.net.load_state_dict(dict['network'])
        self.steps_taken = dict['step']

        logging.info('Resuming from step: %d' % (self.steps_taken))

    def save(self):
        assert hasattr(
            self, 'optimizer'), "Only Trainers should be saving the model, other instances should not be."

        save_path = os.path.join(self.save_dir, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_dict = {
            'optimizer': self.optimizer.state_dict(),
            'network': self.net.state_dict(),
            'lr': self.learning_rate,
            'step': self.steps_taken,
            'np_random_state': np.random.get_state(),
            'py_random_state': random.getstate(),
            'torch_random_state': torch.get_rng_state()
        }

        torch.save(save_dict, os.path.join(save_path, 'model.pth'))
