from __future__ import print_function
from __future__ import absolute_import
from dfp.cfg import EXP_MODE, CFG, VARIABLES, CFG_Base, CFG_Components
from dfp.cfg.Experiment import Experiment
import os

import logging

cfg_vision = {
    VARIABLES.labels: CFG.gt,
}

cfg_automap = {
}

cfg_automap_labels = {
}

#EXP_MODE.visualize, train, evaluate

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-fs', '--force-save', action='store_true',
    #                     help="Force Save the current experiment, regardless of whether there is already a configuration. This will overwrite existing configurations, so becareful with how it is used.")
    # parser.add_argument('-fl', '--force-load', action='store_true',
    #                     help="Force Load the current experiment, regardless of whether the commit hash matches or not. Prevents a reversion before loading the model. The model configuration at this state will not be saved.")
    # parser.add_argument('-d', '--debug', action='store_true',
    #                     help='Turn on debugging mode. Force print everything to STDOUT, and in addition, visualize the inputs.')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the current experiment. Must be in <experiment type>-<name> format')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to config file (doom)')
    args = parser.parse_args()
    # TODO: Visualization argument. Easy debugging

    e = Experiment(mode=EXP_MODE.train, exp_name=args.name, cfg=args.config, use_gpu=True, cfg_vision=cfg_vision, cfg_automap=cfg_automap, cfg_automap_labels=cfg_automap_labels,
                   debug=False)

    if not e.load():
        e.save()

    runner = e.fetch_exp()
    r = runner(e)
    r.run()
