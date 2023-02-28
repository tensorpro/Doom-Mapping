from __future__ import print_function
from __future__ import absolute_import

from dfp.cfg import EXP_MODE, CFG, VARIABLES, CFG_Base, CFG_Components, FORCE
from dfp.cfg.Experiment import Experiment
import os

import logging

cfg_vision = {
    VARIABLES.automap: CFG.gt,
    VARIABLES.grey_img: CFG.gt
}

cfg_automap = {
}

cfg_automap_labels = {
    VARIABLES.depth: CFG.gt,
    VARIABLES.labels: CFG.gt,
    VARIABLES.motion: CFG.gt
}

#EXP_MODE.visualize, train, evaluate
e = Experiment(mode=EXP_MODE.visualize, exp_name="d3_map-img+depth+label+map", cfg="./scenarios/D3_battle.cfg", use_gpu=False,
               cfg_vision=cfg_vision, cfg_automap=cfg_automap, cfg_automap_labels=cfg_automap_labels,
               debug=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', '--force-save', action='store_true',
                        help="Force Save the current experiment, regardless of whether there is already a configuration. This will overwrite existing configurations, so becareful with how it is used.")
    parser.add_argument('-fl', '--force-load', action='store_true',
                        help="Force Load the current experiment, regardless of whether the commit hash matches or not. Prevents a reversion before loading the model. The model configuration at this state will not be saved.")
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Turn on debugging mode. Force print everything to STDOUT, and in addition, visualize the inputs.')
    # TODO: Visualization argument. Easy debugging

    if not e.load(force=FORCE.ignore_commit):
        e.save()

    runner = e.fetch_exp()
    runner(e).run(10)
