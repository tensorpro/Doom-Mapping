from __future__ import print_function, absolute_import

from .enums import EXP_MODE, CFG, CFG_Components, FORCE, VARIABLES
from ..runner.evaluator import Evaluator
from ..runner.trainer import Trainer
from ..runner.visualizer import Visualizer
from ..runner.generalizer import Generalizer

import os
import re
import logging
import subprocess as sp
import json

from ast import literal_eval as make_tuple

exp_filter = re.compile('^(.*)-(.*)$', re.IGNORECASE)


def split(str):
    m = exp_filter.match(str)

    assert m is not None, "%s is an invalid name, must be prefixed by experiment name" % (str)
    return m.group(1), m.group(2)


class Experiment(object):
    def __init__(self, mode, exp_name, cfg, img_res=(84, 84), map_res=(84, 84),
                 map_history=16, model_dir="models", use_gpu=True,
                 cfg_vision=None, cfg_automap=None, cfg_automap_labels=None,
                 debug=False, pred_net_loader=None, invuln=False, eval_comp=True):
        assert map_res[0] == map_res[1], "Map resolution must be square"
        # Debug will override and make the logging level global.
        # It will also enable some output information in the lower levels [WIP]
        self.debug = debug
        # Invinciblility for debugging
        self.invuln = invuln

        # Resolve the experiment logistics
        exp, self.name = split(exp_name)
        self.save_dir = os.path.join(model_dir, exp)
        self.save_path = os.path.join(model_dir, exp, self.name)

        # Make directories before proceeding
        self.makedirs()

        # Additional Parameters
        self.use_gpu = use_gpu
        self.exp_mode = EXP_MODE(mode)
        self.vd_cfg = cfg
        self.num_agents = 8  # this stays constant
        self.fetch_exp()  # use this to configure the logging level

        if not self.use_gpu:
            print("WARNING: The GPU is not used. Ignore if intended.")
            logging.warn("WARNING: The GPU is not used. Ignore if intended.")

        # Resolve the Options used
        if eval_comp:
            self.components = CFG_Components(
                cfg_vision=cfg_vision, cfg_automap=cfg_automap, cfg_automap_labels=cfg_automap_labels)

            if self.components.use_automap_net:
                assert map_history > 0, "Map history frames must be greater than 0, if automap is used"
                self.map_history = map_history
                logging.info('Using a mapping history of %d' % (self.map_history))
            else:
                self.map_history = None

            if self.components.save_img or self.components.save_labels or self.components.save_depth:
                self.img_shape = (self.components.vision_channels,) + img_res
            else:
                self.img_shape = None

            if self.components.save_automap or self.components.save_automap_labels:
                self.map_shape = (self.components.automap_channels,) + map_res
            else:
                self.map_shape = None

            # TODO: Work on pred_net_loader configuration too
            if pred_net_loader:
                logging.warn("Pred Net Loader is not currently implemented. Proceed with caution")

    def makedirs(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def fetch_exp(self):
        if self.exp_mode == EXP_MODE.train:
            if self.debug:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s %(message)s',
                                    datefmt='%m/%d/%Y %I:%M:%S %p')
            else:
                logging.basicConfig(filename=os.path.join(self.save_path,
                                                          'info.log'),
                                    level=logging.INFO,
                                    format='%(asctime)s %(message)s',
                                    datefmt='%m/%d/%Y %I:%M:%S %p')
            return Trainer

        if self.exp_mode == EXP_MODE.inference:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
            return Evaluator

        if self.exp_mode == EXP_MODE.visualize:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
            return Visualizer

        if self.exp_mode == EXP_MODE.generalize:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
            return Generalizer

    def save(self, force=False):
        cmd = "git rev-parse --verify HEAD".split()
        commit_hash = sp.check_output(cmd)

        logging.info("Commit Hash: %s" % (commit_hash))

        save_map = {
            'commit_hash': commit_hash,
            'vd_cfg': self.vd_cfg,
            'components': self.components.serialize(),
            'img_shape': str(self.img_shape),
            'map_shape': str(self.map_shape),
            'map_history': str(self.map_history),
            # TODO: Add the name of the prediction network here, or blank if it doesn't exist
            'pred_net_name': ''
        }

        path = os.path.join(self.save_dir, self.name, 'exp_config.json')

        if not force and os.path.exists(path):
            logging.info("Save path already exists. Refusing to save.")
            return

        if force:
            logging.warn(
                "Force save is enabled. Will overwrite current configuration with given parameters")
        # if not os.path.exists(path):
        logging.info(
            'Could not find the exp_config.json file. Generating one with the current arguments given.')
        # save the experiment configuration
        with open(path, 'w') as f:
            json.dump(save_map, f, sort_keys=True, indent=4, separators=(',', ': '))
            f.close()

    def load(self, force=FORCE.default):
        cmd = "git rev-parse --verify HEAD".split()
        commit_hash = sp.check_output(cmd)

        path = os.path.join(self.save_dir, self.name, 'exp_config.json')

        if os.path.exists(path):
            logging.info(
                'Found the exp_config.json file. Ignoring the given arguments, and using the file instead.')
            # load the experiment configuration, ignore defaults

            # Load only python strings, instead of unicode
            def str_hook(obj):
                return {k.encode('utf-8') if isinstance(k, unicode) else k:
                        v.encode('utf-8') if isinstance(v, unicode) else v
                        for k, v in obj}

            with open(path, 'r') as f:
                d = json.load(f, object_pairs_hook=str_hook)
                f.close()

            # Assign the values that weren't None
            correct_commit_hash = d['commit_hash']
            if force == FORCE.ignore_all:
                logging.warn(
                    "Attempting to use the current commit hash, since force has been enabled")
            elif force == FORCE.ignore_commit:
                logging.warn(
                    "Ignoring commit, but still loading configuration."
                )
            # NOTE: Rest of the branches assume FORCE.default to be set.
            elif commit_hash != correct_commit_hash:
                logging.warn("Commit hashes are different. Will revert and retry")
                cmd = "git checkout %s" % (correct_commit_hash)
                try:
                    out = sp.check_output(cmd.split())
                    # NOTE: Re-checkout the Experiment file
                    cmd = 'git checkout master %s' % ('dfp/cfg/Experiment.py')
                    out = sp.check_output(cmd.split())

                    logging.info("Successfully Reverted. Run the command again to proceed")
                    return True

                except sp.CalledProcessError as e:
                    logging.warn(
                        "If the current commit hash should work, try enabling the force parameter")
                    logging.warn(
                        "Otherwise, make sure the worktree is clean")
                    exit(1)  # don't run the experiment at all
            else:
                logging.info("Commit Hash matches, loading components")

            self.vd_cfg = d['vd_cfg']
            self.components = CFG_Components.deserialize(d['components'])
            self.pred_net_name = d['pred_net_name']
            if make_tuple(d['img_shape']) is not None:
                self.img_shape = tuple(map(lambda x: int(x), list(make_tuple(d['img_shape']))))
            else:
                self.img_shape = None

            if make_tuple(d['map_shape']) is not None:
                self.map_shape = tuple(map(lambda x: int(x), list(make_tuple(d['map_shape']))))
                if d.get('map_history'):
                    self.map_history = int(d['map_history'])
                else:
                    self.map_history = None
            else:
                self.map_shape = None
                self.map_history = None

            return True
        else:
            logging.warn(
                'Could not find the exp_config.txt file. Behavior is undefined, unless config is provided.')
            return False
