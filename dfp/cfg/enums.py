from enum import Enum
from collections import defaultdict
import logging
from ..global_utils import pp


# Represent each configuration as an enum...
# Use CFG_ as a prefix to declare that it is a configuration item
from enum import EnumMeta


class DefaultEnumMeta(EnumMeta):
    default = object()

    def __call__(cls, value=default, *args, **kwargs):
        if value is DefaultEnumMeta.default:
            # Assume the first enum is default
            return next(iter(cls))
        return super(DefaultEnumMeta, cls).__call__(value, *args, **kwargs)


class EXP_MODE(Enum):
    """
    Configuration for possible modes to run the DFP in.
    train -- train the DFP. Run inferences every so often to track progress
    inference -- run a single inference, or 55k steps and record the result
    visualize -- for a minute, run the DFP and see what the agent also sees.
                 Useful for debugging or evaluating the current policy
    """
    __metaclass__ = DefaultEnumMeta
    __order__ = 'train inference visualize generalize'
    train = 0
    inference = 1
    visualize = 2
    generalize = 3


class CFG(Enum):
    """
    Configuratino for each item. They can be either none (no input),
    ground truth (gt), or predicted (pt)
    """
    __metaclass__ = DefaultEnumMeta
    __order__ = 'none gt pred gt_enemies pred_enemies'
    none = 0
    gt = 1
    pred = 2
    # Hacky, but specific things for automap labels
    gt_enemies = 3
    pred_enemies = 4
    doom_provided = 5


class VARIABLES(Enum):
    """
    The inputs to various components of the DFP.
    """
    __metaclass__ = DefaultEnumMeta
    __order__ = 'grey_img rgb_img depth labels motion automap'
    # Images
    grey_img = 0
    rgb_img = 1

    depth = 2
    labels = 3
    motion = 4

    automap = 5
    automap_labels = 6

    measurements = 7


class FORCE(Enum):
    """
    The inputs to various components of the DFP.
    """
    __metaclass__ = DefaultEnumMeta
    __order__ = 'default ignore_commit ignore_all'

    # Force parameters
    default = 0
    # default reverts the program to the last valid commit,
    # and uses the full configuration file stored
    ignore_commit = 1
    # ignore commit does not revert, but uses the configuration that is stored
    ignore_all = 2
    # ignore_all does not revert, and ignores the commit that is used.


class CFG_Base(object):
    def __init__(self, kwargs=None):
        self.config = defaultdict(CFG)
        # kwargs is None, when the component isn't enabled.
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                assert isinstance(k, VARIABLES), "Argument must be a valid VARIABLES enum"
                assert isinstance(v, CFG), "Argument must be a valid CFG enum"
                # Another sanity check
                self.config[k] = v

    def __getitem__(self, keys):
        if type(keys) is list:
            return [self.config[k] for k in keys]
        return self.config[keys]

    def iteritems(self):
        return self.config.iteritems()

    def iterkeys(self):
        return self.config.iterkeys()

    def itervalues(self):
        return self.config.itervalues()

    def is_none(self):
        """
        Check if the current cfg_base only has none
        """
        for k, v in self.config.iteritems():
            if v != CFG.none:
                return False

        if len(self.config) == 0:
            return False
        else:
            return True

    def to_dep(self):
        """
        Convert the current dependecy object into pairs of (variable, configruation)
        """
        vars = filter(lambda x: self.config[x] != CFG.none, list(VARIABLES))
        return map(lambda x: (x, self.config[x]), vars)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return pp.pformat(self.config)

    def serialize(self):
        return {k.name: v.name for k, v in self.config.iteritems()}

    @classmethod
    def deserialize(self, d):
        obj = CFG_Base({getattr(VARIABLES, k): getattr(CFG, v)
                        for k, v in d.iteritems()})
        # Return None, if everything is None. Used in CFG_Components
        return None if obj.is_none() else obj


class CFG_Components(object):
    def __init__(self, cfg_vision=None, cfg_automap=None, cfg_automap_labels=None, cfg_pred_net=None):
        self.cfg_vision = CFG_Base(cfg_vision)
        self.cfg_automap = CFG_Base() if cfg_automap is None else CFG_Base(cfg_automap)
        self.cfg_automap_labels = CFG_Base() if cfg_automap_labels is None else CFG_Base(cfg_automap_labels)

        # Check for circular dependencies
        assert self.cfg_vision[VARIABLES.rgb_img] == CFG.none, "Vision cannot depend on rgb_img"

        assert self.cfg_automap[VARIABLES.automap] == CFG.none, "Automap cannot depend on automap"
        assert self.cfg_automap_labels[VARIABLES.automap] == CFG.none, "Automap Labels cannot depend on automap"

        for m in [self.cfg_vision, self.cfg_automap]:
            if m[VARIABLES.labels] in [CFG.pred_enemies, CFG.gt_enemies]:
                raise ValueError, "Invalid VARIABLES to CFG mapping: cannot use gt_enemies or pred_enemies in anything other than automap_labels"
        self.compute_dep()

    def check_dep(self, var, cfg):
        assert isinstance(var, VARIABLES), "To check for a dependency, var must be a VARIABLE enum"
        assert isinstance(cfg, CFG), "To check for a dependency, cfg must be a CFG enum"

        return (self.cfg_automap[var] == cfg or
                self.cfg_automap_labels[var] == cfg or
                self.cfg_vision[var] == cfg)

    def compute_dep(self):
        # NOTE: Calculate dependencies for individual components
        self.use_pred_depth = self.check_dep(VARIABLES.depth, CFG.pred)
        self.use_pred_labels = (self.check_dep(VARIABLES.labels, CFG.pred) or
                                self.check_dep(VARIABLES.labels, CFG.pred_enemies))
        self.use_pred_motion = self.check_dep(VARIABLES.motion, CFG.pred)

        self.use_gt_depth = self.check_dep(VARIABLES.depth, CFG.gt)
        self.use_gt_labels = (self.check_dep(VARIABLES.labels, CFG.gt) or
                              self.check_dep(VARIABLES.labels, CFG.gt_enemies))
        self.use_gt_motion = self.check_dep(VARIABLES.motion, CFG.gt)
        # NOTE: Automap Specific things
        self.use_doom_automap = (self.cfg_vision[VARIABLES.automap] == CFG.doom_provided or
                                 self.cfg_vision[VARIABLES.automap_labels] == CFG.doom_provided)
        self.use_calc_automap = ((self.cfg_vision[VARIABLES.automap] != CFG.doom_provided and
                                  self.cfg_vision[VARIABLES.automap] != CFG.none) or
                                 (self.cfg_vision[VARIABLES.automap_labels] != CFG.doom_provided and
                                  self.cfg_vision[VARIABLES.automap_labels] != CFG.none))

        # Check if individual networks are used
        self.use_automap_net = (self.cfg_vision[VARIABLES.automap] != CFG.none or
                                self.cfg_vision[VARIABLES.automap_labels] != CFG.none)
        self.use_vision_net = (self.cfg_vision[VARIABLES.depth] != CFG.none or
                               self.cfg_vision[VARIABLES.labels] != CFG.none or
                               self.cfg_vision[VARIABLES.grey_img] != CFG.none)

        # Prediction net parameter for compatibility
        self.use_pred_nets = (self.use_pred_depth or self.use_pred_labels or self.use_pred_motion)

        # NOTE: Some Sanity check assertions
        assert self.use_automap_net or self.use_vision_net, "Neither the Vision Network or Automap Network was enabled."
        if self.use_doom_automap:
            assert not (self.use_gt_motion or
                        self.use_pred_motion), "Enabling Doom Automaps and Motion is unnecessary"
            assert not self.use_calc_automap, "Enabling both doom and calcuated automap is unnecessary"

        if self.use_calc_automap:
            assert (self.use_gt_motion or
                    self.use_pred_motion), "Motion must be enabled for automap creation"
            assert not (self.use_doom_automap), 'Doom automap cannot be enabled for calculated automaps'

        if self.use_automap_net:
            assert self.use_doom_automap or self.use_calc_automap, "Must include some kind of doom map in the vision dependency"
        else:
            assert not (
                self.use_doom_automap or self.use_calc_automap), "Cannot include mapping, if network does not use it"

        if self.use_calc_automap:
            assert (self.cfg_automap[VARIABLES.depth] != CFG.none or
                    self.cfg_automap_labels[VARIABLES.depth] != CFG.none), "Calculated Automap Needs some form of depth"
        # This assumption might change in the future, but for now this is certainly the case.
        assert not (
            self.use_gt_motion and self.use_pred_motion), "Enabling both predicted and gt motion is unnecessaary"

        # Check what needs to be saved in the replay buffer
        self.save_img = self.cfg_vision[VARIABLES.grey_img] != CFG.none
        self.save_depth = self.cfg_vision[VARIABLES.depth] != CFG.none
        self.save_labels = self.cfg_vision[VARIABLES.labels] != CFG.none
        # TODO: Might become an issue if automap is enabled, but the automap labels is not
        # FIXME: In addition, there might be some nasty bugs w/ enabling the doom automap...
        self.save_automap = self.use_automap_net and (
            self.cfg_automap[VARIABLES.depth] != CFG.none or self.use_doom_automap)
        self.save_automap_labels = self.use_automap_net and self.cfg_automap_labels[
            VARIABLES.labels] != CFG.none

        # Some logging for the user to see
        logging.info("Successfully calculated required Variables")
        # Print out information
        COMPONENTS = ""
        if self.use_pred_depth:
            COMPONENTS += " PRED_DEPTH"

        if self.use_pred_labels:
            COMPONENTS += " PRED_LABELS"

        if self.use_pred_motion:
            COMPONENTS += " PRED_MOTION"

        if self.use_calc_automap:
            COMPONENTS += " CALC_AUTOMAP"

        if self.use_gt_depth:
            COMPONENTS += " GT_DEPTH"

        if self.use_gt_labels:
            COMPONENTS += " GT_LABELS"

        if self.use_gt_motion:
            COMPONENTS += " GT_MOTION"

        if self.use_doom_automap:
            COMPONENTS += " DOOM_AUTOMAP"

        if len(COMPONENTS) > 0:
            logging.info("Components required:%s" % (COMPONENTS))
        else:
            logging.info("No components discovered")

        NETWORKS = ""
        if self.use_automap_net:
            NETWORKS += " AUTOMAP_NET"

        if self.use_vision_net:
            NETWORKS += " VISION_NET"

        if len(NETWORKS) > 0:
            logging.info("Networks required:%s" % (NETWORKS))
        else:
            logging.warn("Networks not discovered, make sure it is configured right")
            logging.warn(
                "This could be because the current initialization has not loaded the configuration yet")

        REPLAY = ""
        if self.save_img:
            REPLAY += " GRAY"

        if self.save_depth:
            REPLAY += " DEPTH"

        if self.save_labels:
            REPLAY += " LABELS"

        if self.save_automap:
            REPLAY += " AUTOMAP"

        if self.save_automap_labels:
            REPLAY += " AUTOMAP_LABELS"

        if len(REPLAY) > 0:
            logging.info("Replay buffers required:%s" % (REPLAY))
        else:
            logging.warn("There is no specific replay buffer required")
            logging.warn(
                "This could be because the current initialization has not loaded the configuration yet")

        # Compute the number of channels required
        self.vision_channels = 0
        if self.use_vision_net:
            if self.cfg_vision[VARIABLES.grey_img] != CFG.none:
                self.vision_channels += 1
            if self.cfg_vision[VARIABLES.depth] != CFG.none:
                self.vision_channels += 1
            if self.cfg_vision[VARIABLES.labels] != CFG.none:
                self.vision_channels += 6

        logging.info("Determined the number of channels required for vision: %d" %
                     (self.vision_channels))

        self.automap_channels = 0
        if self.use_calc_automap:
            if (self.cfg_automap[VARIABLES.depth] != CFG.none):
                # TODO: This is not a great implementation, it assumes the automap is used to turn on/off the depth
                # ideally we'd like to merge but don't have time to think of a good way now
                self.automap_channels += 1
            if (self.cfg_automap[VARIABLES.labels] != CFG.none or
                    self.cfg_automap_labels[VARIABLES.labels] != CFG.none):
                if self.cfg_automap_labels[VARIABLES.labels] in [CFG.gt_enemies, CFG.pred_enemies]:
                    self.automap_channels += 1
                else:
                    # NOTE: uses all the labels and droppings
                    self.automap_channels += (6 + 1)

        elif self.use_doom_automap:
            self.automap_channels += 1

        logging.info("Determined the number of channels required for automap: %d" %
                     (self.automap_channels))

    def serialize(self):
        return {
            "cfg_vision": self.cfg_vision.serialize(),
            "cfg_automap": self.cfg_automap.serialize(),
            "cfg_automap_labels": self.cfg_automap_labels.serialize()
        }

    @classmethod
    def deserialize(self, d):
        cfg_vision = CFG_Base.deserialize(d["cfg_vision"])
        cfg_automap = CFG_Base.deserialize(d['cfg_automap'])
        cfg_automap_labels = CFG_Base.deserialize(d['cfg_automap_labels'])

        return CFG_Components(cfg_vision=cfg_vision,
                              cfg_automap=cfg_automap,
                              cfg_automap_labels=cfg_automap_labels)
