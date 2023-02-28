from __future__ import print_function
from __future__ import absolute_import

from ..memory.memory import Memory
from .base.multi_runner import MultiRunner
from ..simulator.DoomSimulator import DoomSimulator
from ..mapping import multimapper

import numpy as np
import logging
import os

import gc


class Evaluator(MultiRunner):
    """
    Evaluate the current network only.
    Was called "Inference" in the original DFP
    """

    def __init__(self, experiment_cfg,
                 test_capacity=55000, future_steps=np.array([1, 2, 4, 8, 16, 32]),
                 min_meas_num=3, batch_size=64, mapper=None):

        if (mapper is None and
            experiment_cfg.components.use_automap_net and
                (not experiment_cfg.components.use_doom_automap)):
            mapper = multimapper.MultiMapper(experiment_cfg)
        else:
            mapper = None

        # Temporary Simulator to parse config file
        simulator = DoomSimulator(experiment_cfg)

        # Initialize Memory before anything else
        memory = Memory(experiment_cfg, simulator.num_available_variables,
                        simulator.num_available_buttons, future_steps,
                        min_meas_num, capacity=test_capacity)

        super(Evaluator, self).__init__(experiment_cfg,
                                        future_steps=future_steps,
                                        batch_size=batch_size, memory=memory,
                                        mapper=mapper)

        self.test_memory = memory
        self.memory = memory

    def test(self, stdout=False):
        """
        Play game with no epsilon control.
        Logs in log_dir
        """
        temp_memory = None
        if self.memory is not self.test_memory:
            # self.memory.prev_term()
            temp_memory = self.memory
            self.swap_memory(self.test_memory)

        self.multiagent.reset()

        # Fill memory buffer, using current policy
        self.multiagent.fill(self.steps_taken, inference=True)

        gc.collect()

        if temp_memory is not None:
            self.swap_memory(temp_memory)
            # self.multiagent.reset()
            # self.memory.skip_current_state()

    def run(self):
        self.test(stdout=True)
