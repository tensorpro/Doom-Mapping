from __future__ import print_function
from __future__ import absolute_import

from .runner import Runner
from ...agents.multiagent import MultiAgent
from ...mapping import multimapper

import numpy as np


class MultiRunner(Runner):
    """
    Class for running MultiAgents.
    See Trainer.py and Inference.py for details.
    """

    def __init__(self, experiment_cfg,
                 future_steps=np.array([1, 2, 4, 8, 16, 32]),
                 batch_size=64, memory=None, mapper=None):

        if (mapper is None and
            experiment_cfg.components.use_automap_net and
                (not experiment_cfg.components.use_doom_automap)):
            mapper = multimapper.MultiMapper(experiment_cfg)
        else:
            mapper = None

        super(MultiRunner, self).__init__(experiment_cfg, future_steps=future_steps)
        # Steps and Agent count
        self.num_agents = experiment_cfg.num_agents

        # Memory
        self.memory = memory

        # Multiagent Initialization
        self.multiagent = MultiAgent(experiment_cfg, self.net,
                                     self.memory, mapper=mapper)

        self.multiagent.init_simulators(experiment_cfg)

    def swap_memory(self, memory):
        assert memory.capacity % self.num_agents == 0, "The capacity should be divisible by threads"

        self.multiagent.swap_memory(memory)
        self.memory = memory
