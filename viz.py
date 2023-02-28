from __future__ import print_function
from __future__ import absolute_import

from dfp import MultiAgent

import os

MODEL_OUT = 'test_d2_navigation.pth'
LOG_OUT = 'test_d2_navigation.txt'

ag = MultiAgent([1], test_window_visible=True, viz_mode=True)

start = ag.load(os.path.join('models', MODEL_OUT, MODEL_OUT),
                cuda_to_cpu=True)
print("starting from %d" % (start))

ag.visualize()
