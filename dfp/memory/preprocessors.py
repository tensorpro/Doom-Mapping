from __future__ import print_function
from __future__ import absolute_import

from ..img_utils.utils import one_hot_encode_batch

import numpy as np

FUTURE_TARGET_LEN = 6

TARGET_SCALE_HEALTH = np.expand_dims((np.expand_dims(
    np.array([30.]), 1) * np.ones((1, FUTURE_TARGET_LEN))).flatten(), 0)
TARGET_SCALE_AMMO_HEALTH_FRAG = np.expand_dims((np.expand_dims(
    np.array([7.5, 30.0, 1.0]), 1) * np.ones((1, FUTURE_TARGET_LEN))).flatten(), 0)

NUM_CHANNEL = 1
NUM_LABELS = 6
NUM_DEPTHS = 1


class Preprocessors(object):
    """
    Defines a set of custom preprocessors.
    """
    @classmethod
    def preprocess_images(self, images, depths, labels):
        if images is not None and depths is not None and labels is not None:
            images = images / 255.0 - 0.5
            depths = depths / 255.0 - 0.5
            labels = one_hot_encode_batch(labels[:, 0], 6)
            return np.concatenate((images, depths, labels.transpose(0, 3, 1, 2)), 1)
        elif images is not None and depths is not None:
            images = images / 255.0 - 0.5
            depths = depths / 255.0 - 0.5
            return np.concatenate((images, depths), 1)
        elif images is not None and labels is not None:
            images = images / 255.0 - 0.5
            labels = one_hot_encode_batch(labels[:, 0], 6)
            return np.concatenate((images, labels.transpose(0, 3, 1, 2)), 1)
        elif depths is not None and labels is not None:
            depths = depths / 255.0 - 0.5
            labels = one_hot_encode_batch(labels[:, 0], 6)
            return np.concatenate((depths, labels.transpose(0, 3, 1, 2)), 1)
        elif images is not None:
            return images / 255.0 - 0.5
        elif depths is not None:
            return depths / 255.0 - 0.5
        elif labels is not None:
            return one_hot_encode_batch(labels[:, 0], 6).transpose(0, 3, 1, 2)
        else:
            return None

    @classmethod
    def preprocess_automaps(self, raw_automap, raw_automap_labels, only_one=False):
        if only_one:
            # Special mode for debugging. Only when we do the enemies only experiment
            return (raw_automap_labels == 1).astype(np.uint8)

        if raw_automap is not None and raw_automap_labels is not None:
            raw_automap = raw_automap / 255.0 - 0.5
            raw_automap_labels = raw_automap_labels[:, 0]
            raw_automap_labels = one_hot_encode_batch(raw_automap_labels, 7).transpose(0, 3, 1, 2)
            return np.concatenate((raw_automap, raw_automap_labels), 1)
        elif raw_automap is not None:
            return raw_automap / 255.0 - 0.5
        elif raw_automap_labels is not None:
            raw_automap_labels = raw_automap_labels[:, 0]
            raw_automap_labels = one_hot_encode_batch(raw_automap_labels, 7).transpose(0, 3, 1, 2)
            return raw_automap_labels
        else:
            return None

    @classmethod
    def preprocess_measurements(self, raw_measurements):
        return (raw_measurements / 100.0) - 0.5

    @classmethod
    def preprocess_targets(self, raw_targets):
        if raw_targets.shape[-1] / 6 == 1:
            # Only health
            return raw_targets / TARGET_SCALE_HEALTH
        elif raw_targets.shape[-1] / 6 == 3:
            return raw_targets / TARGET_SCALE_AMMO_HEALTH_FRAG
        else:
            raise NotImplementedError, "Targets must have either 1 or 3 dims"

    @classmethod
    def preprocess_rewards(self, raw_rewards):
        return raw_rewards

    @classmethod
    def preprocess_terminals(self, raw_terminals):
        return raw_terminals

    @classmethod
    def preprocess_actions(self, raw_action):
        return raw_action


class PreNet(Preprocessors):
    """
    Defines a set of custom preprocessors.
    """
    @classmethod
    def preprocess_images(self, images, depths, labels):
        images = images / 255.0 - 0.5

        if depths is not None and labels is not None:
            depths = depths / 255.0 - 0.5
            # 7 channels
            labels = one_hot_encode_batch(labels[:, 0], 6)
            imgs = np.concatenate((images, depths, labels.transpose(0, 3, 1, 2)), 1)
        elif depths is not None:
            depths = depths / 255.0 - 0.5
            imgs = np.concatenate((images, depths), 1)
        elif labels is not None:
            # 7 channels
            labels = one_hot_encode_batch(labels[:, 0], 6)
            imgs = np.concatenate((images, labels.transpose(0, 3, 1, 2)), 1)
        else:
            imgs = images

        return imgs

    @classmethod
    def preprocess_automaps(self, raw_automap, raw_automap_labels, eps=1e-6):
        if raw_automap_labels is None:
            return raw_automap / 255.0 - 0.5
        elif raw_automap_labels is not None:
            raw_automap = raw_automap / 255.0 - 0.5
            raw_automap_labels = one_hot_encode_batch(raw_automap_labels[:, 0], 6)
            return np.concatenate((raw_automap, raw_automap_labels.transpose(0, 3, 1, 2)), 1)
