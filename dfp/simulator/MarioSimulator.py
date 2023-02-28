# Thin wrapper around the gym mario implementation
import gym
# Small hack on my local computers to get it to recognize that the mario exists
import ppaquette_gym_super_mario

import cv2
import numpy as np


class MarioSimulator(object):
    def __init__(self, resolution=(56, 64), frame_skip=4):
        self.env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
        # no frameskip so we'll have to augment on our own...

        self.num_available_buttons = 6  # hard coded (sad)
        self.num_available_variables = 5  # exclude life and level

        self.frame_skip = frame_skip
        self.resolution = resolution

    def stack_vars(self, vars):
        """
        Take the given variables and stack them into a single vector
        """
        # The "life" variable is excluded because the episode is made to
        # reset every time.
        meas = np.array([vars['coins'], vars['distance'],
                         vars['player_status'], vars['score'], 1 - vars['time'] / 400.0])

        # subtract from 1 to "maximize" the time not spent completing the level, thereby
        # minimizing the total amount of time spent

        return meas

    def init(self):
        self.env.render()
        self.env.reset()  # throw away the first frame, cause it doesn't have rewards or any kind of state

    def close(self):
        self.env.close()

    def make_action(self, index):
        """
        Gets the 'state' the way DFP presents it.
        Returns image of size resolution, a vector of measurements, and whether the state is terminal or not
        DFP also assumes that only one button is pressed. It might affect the performance in Mario
        """
        # Expected input is a one-hot encoding
        action_list = [0 for _ in range(self.num_available_buttons)]
        action_list[index] = 1

        # simulate frameskipping
        for _ in range(self.frame_skip):
            state = self.env.step(action_list)
            if state[2]:  # terminal state
                break

        ################
        # Image Buffer #
        ################
        image = cv2.cvtColor(state[0], cv2.COLOR_BGR2GRAY)
        image = cv2.resize(
            image, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_LINEAR)

        ###########
        # Rewards #
        ###########
        reward = state[1]

        ######################
        # Measurement Stream #
        ######################
        measurements = self.stack_vars(state[-1])

        ##################
        # Terminal State #
        ##################
        is_terminal = state[2]

        if is_terminal:
            # prepare for the next episode
            self.env.reset()

        return image, measurements, reward, is_terminal
