from __future__ import print_function
from __future__ import absolute_import
from dfp.cfg import EXP_MODE, CFG, VARIABLES, CFG_Base, CFG_Components
from dfp.cfg.Experiment import Experiment
import os

import logging

cfg_vision = {
    VARIABLES.depth: CFG.gt,
    VARIABLES.labels: CFG.gt,
    VARIABLES.automap: CFG.gt
}

cfg_automap = {
}

cfg_automap_labels = {
    VARIABLES.depth: CFG.gt,
    VARIABLES.labels: CFG.gt,
    VARIABLES.motion: CFG.gt
}

#EXP_MODE.visualize, train, evaluate

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the current experiment. Must be in <experiment type>-<name> format')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to config file (doom)')
    args = parser.parse_args()
    # TODO: Visualization argument. Easy debugging

    e = Experiment(mode=EXP_MODE.train, exp_name=args.name, cfg=args.config, use_gpu=True, cfg_vision=cfg_vision,
                   cfg_automap=cfg_automap, cfg_automap_labels=cfg_automap_labels, map_history=32, debug=False)

    if not e.load():
        e.save()

    runner = e.fetch_exp()
    r = runner(e)
    r.run()
