"""Wrapper to make the a1 environment suitable for OpenAI gym."""
from unittest import result
import gym

from envs import env_builder

class A1GymEnv(gym.Env):
  """A1 environment that supports the gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self, args, enable_rendering):

    self._env = env_builder.build_regular_env(args, enable_rendering)

    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

    self.args = args
    self.enable_rendering = enable_rendering

  def step(self, action):
    results = self._env.step(action)

    foot_pos = self._env.robot.GetFootPositionsInBaseFrame()
    return results

  def reset(self):
    return self._env.reset()

  def close(self):
    self._env.close()

  def render(self, mode):
    return self._env.render(mode)

  def __getattr__(self, attr):
    return getattr(self._env, attr)
