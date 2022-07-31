
"""Simple openloop trajectory generators."""
import attr
from gym import spaces
import numpy as np

from robots import a1_pose_utils
from locomotion.robots import laikago_pose_utils


ACTION_LIMIT_LOW = (-0.802851455917, -1.0471975512, -2.69653369433)
ACTION_LIMIT_HIGH  = (0.802851455917, 4.18879020479, -0.916297857297)

# Limits for laikago default position, overriding the limits in urdf
ACTION_LIMIT_LOW = (-0.802851455917, -1.71719755, -1.446533694)
ACTION_LIMIT_HIGH  = (0.802851455917, 3.518790205, 0.333702142)

class A1PoseOffsetGenerator(object):
  """A trajectory generator that return constant motor angles."""
  def __init__(
      self,
      init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
      init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
      init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE,
      action_high=ACTION_LIMIT_HIGH,
      action_low=ACTION_LIMIT_LOW
  ):
    """Initializes the controller.
    Args:
      action_limit: a tuple of [limit_abduction, limit_hip, limit_knee]
    """
    self._pose = np.array(
        attr.astuple(
            laikago_pose_utils.LaikagoPose(abduction_angle_0=init_abduction,
                                           hip_angle_0=init_hip,
                                           knee_angle_0=init_knee,
                                           abduction_angle_1=init_abduction,
                                           hip_angle_1=init_hip,
                                           knee_angle_1=init_knee,
                                           abduction_angle_2=init_abduction,
                                           hip_angle_2=init_hip,
                                           knee_angle_2=init_knee,
                                           abduction_angle_3=init_abduction,
                                           hip_angle_3=init_hip,
                                           knee_angle_3=init_knee)))
    action_high = np.hstack([action_high] * 4)
    action_low = np.hstack([action_low] * 4)
    self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.

    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    return self._pose + input_action

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation
