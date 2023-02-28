import vizdoom
import numpy as np

import logging

from itertools import product


class DoomSimulator(object):

    def __init__(self, experiment_cfg,
                 frame_skip=4, visible=False):
        map = "MAP01"
        self.game = vizdoom.DoomGame()
        # Game settings
        self.frame_skip = frame_skip
        self.game_initialized = False
        self.invuln = experiment_cfg.invuln

        resolution = experiment_cfg.img_shape
        self.img_resolution = None
        if resolution is not None:
            if len(resolution) == 3:
                # first element must be channel
                self.img_resolution = (resolution[1], resolution[2])
            else:
                self.img_resolution = (84, 84)

        resolution = experiment_cfg.map_shape
        self.map_resolution = None
        if resolution is not None:
            if len(resolution) == 3:
                self.map_resolution = (resolution[1], resolution[2])
            else:
                self.map_resolution = (84, 84)

        self.use_gt_depth = experiment_cfg.components.use_gt_depth
        self.use_gt_labels = experiment_cfg.components.use_gt_labels
        self.use_gt_motion = experiment_cfg.components.use_gt_motion

        self.use_pred_nets = experiment_cfg.components.use_pred_nets
        self.use_doom_automap = experiment_cfg.components.use_doom_automap

        options = ''
        if self.use_gt_depth:
            options += ' DEPTH'
        if self.use_gt_labels:
            options += ' LABELS'
        if self.use_gt_motion:
            options += ' MOTION'
        if self.use_doom_automap:
            options += ' AUTOMAP'

        if self.use_pred_nets:
            options += ' RGB_IMAGES'
        else:
            options += ' GRAY_IMAGES'

        if len(options) > 0:
            logging.info('Simulator with options:%s' % (options))

        if self.img_resolution:
            logging.info('Simulator has IMAGE resolution %s' % (str(self.img_resolution)))
        if self.map_resolution:
            logging.info('Simulator has MAP resolution %s' % (str(self.map_resolution)))

        # Map and configuration files
        self.game.load_config(experiment_cfg.vd_cfg)
        self.game.set_doom_map(map)

        self.settings(visible=visible)

    def settings(self, visible=False):
        ###################
        # Screen settings #
        ###################
        try:
            self.game.set_screen_resolution(
                vizdoom.ScreenResolution, "RES_%dX%d" % self.resolution)
            self.resize = False
        except:
            self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_160X120)
            # self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
            self.resize = True

        # Resolve what kind of screen we are using
        if self.use_pred_nets:
            logging.info("Using RGB24 screen format")
            screen_fmt = vizdoom.ScreenFormat.RGB24
        else:
            logging.info("Using GRAY8 screen format")
            screen_fmt = vizdoom.ScreenFormat.GRAY8

        self.game.set_screen_format(screen_fmt)

        self.game.set_window_visible(visible)

        # NOTE: Obtain possible buttons/varaibles
        num_possible_buttons = self.game.get_available_buttons_size()
        self.available_buttons = np.array(
            list(product('01', repeat=num_possible_buttons))).astype(np.bool).tolist()
        self.available_variables = self.game.get_available_game_variables()

        self.num_available_buttons = len(self.available_buttons)
        self.num_available_variables = self.game.get_available_game_variables_size()

        logging.info("Simulator has %d buttons and %d variables" % (self.num_available_buttons,
                                                                    self.num_available_variables))
        # If there is no network, use what is given.
        # Otherwise use the networks as predictions
        if not self.use_pred_nets:
            self.game.set_depth_buffer_enabled(self.use_gt_depth)
            self.game.set_labels_buffer_enabled(self.use_gt_labels)

        if self.use_doom_automap:
            # TODO: Fix other settings for the automap, simpler the better
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(vizdoom.AutomapMode.OBJECTS_WITH_SIZE)
            self.game.set_automap_rotate(True)
            self.game.set_automap_render_textures(False)

        if self.use_gt_motion:
            # Important that this computation goes after the num_available_variables has been set
            # Or this will faultiy return everything, since these parameters count as variables.
            logging.info("Automap variables have been set.")
            for var in [vizdoom.GameVariable.POSITION_X,
                        vizdoom.GameVariable.POSITION_Y,
                        vizdoom.GameVariable.POSITION_Z,
                        vizdoom.GameVariable.ANGLE]:
                self.game.add_available_game_variable(var)

    def init(self):
        if not self.game_initialized:
            self.game.init()
            self.game_initialized = True

        self.game.new_episode()

        if self.invuln:
            self.game.send_game_command('iddqd')

        if self.use_doom_automap:
            self.game.send_game_command('am_scale 0.8')

    def close(self):
        self.game.close()
        self.game_initialized = False

    def reset(self):
        self.init()

    def make_action(self, index):
        """
        Gets the 'state' the way DFP presents it.
        Returns image of size resolution, a vector of measurements, and whether the state is terminal or not
        """
        ###########
        # Rewards #
        ###########
        rwrd = self.game.make_action(self.available_buttons[index],
                                     self.frame_skip)
        term = self.game.is_episode_finished() or self.game.is_player_dead()

        if term:
            # NOTE: This fixes a bug whenever the terminal state has important reward (like simpler_basic.cfg)
            RET = self.get_state(term, rwrd)  # Obtain the state, if it exists

            self.reset()  # in multiplayer multi_simulator takes care of this
            # should ideally put nan here, but since it's an int...

            if RET is not None:
                return RET  # only return state if it existed, otherwise zero out the states

            # resizing done later
            if self.use_pred_nets:
                raw_img = np.zeros((3, 120, 160),
                                   dtype=np.uint8)
            else:
                raw_img = np.zeros((1, 120, 160),
                                   dtype=np.uint8)

            if self.use_gt_depth:
                depth_img = np.zeros((1, 120, 160),
                                     dtype=np.uint8)
            else:
                depth_img = None

            if self.use_gt_labels:
                label_img = np.zeros((1, 120, 160),
                                     dtype=np.uint8)
            else:
                label_img = None

            if self.use_doom_automap:
                automap_img = np.zeros((1, 120, 160),
                                       dtype=np.uint8)
            else:
                automap_img = None

            # should ideally put nan here, but since it's an int...
            meas = np.zeros(self.num_available_variables, dtype=np.uint32)
            if self.use_gt_motion:
                coord = np.zeros(4)
            else:
                coord = None

            return raw_img, depth_img, label_img, meas, coord, rwrd, term, automap_img

        return self.get_state(term, rwrd)

    def get_state(self, term, rwrd):
        state = self.game.get_state()

        if state is None:
            return None

        depth_img = None
        label_img = None
        automap_img = None
        if state is None:
            raw_img = None
            meas = None
        else:
            if self.use_pred_nets:
                # At least one of Depth Labels will be predicted.
                # However, the screen items get returned as RGB images.
                raw_img = state.screen_buffer.transpose(2, 0, 1)
            else:
                # No prediction network, use regular training method
                # Image buffer
                raw_img = np.expand_dims(state.screen_buffer, 0)

            if self.use_gt_depth:
                # both automap and depth requires use of the depth channel
                depth_img = np.expand_dims(state.depth_buffer, 0)

            if self.use_gt_labels:
                label_img = np.expand_dims(state.labels_buffer, 0)

            if self.use_doom_automap:
                # TODO: Check if this expands to the right things
                automap_img = np.expand_dims(state.automap_buffer, 0)

            # this is a numpy array of game variables specified by the scenario
            if self.use_gt_motion:
                meas = state.game_variables[:-4]

                coord = state.game_variables[-4:]
            else:
                meas = state.game_variables
                coord = None

        # NOTE: Hack on the frags, since dfp seems to dislike measurements close to the end of the episode...
        return raw_img, depth_img, label_img, meas, coord, rwrd, term, automap_img
