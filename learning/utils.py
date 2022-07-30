import inspect
import os
from collections import OrderedDict
from sched import scheduler
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Got this from rl-zoo
def lr_schedule(initial_value: Union[float, str], 
                lr_type: str, 
                final_value: Union[float, str] = 1e5, 
                total_timesteps: Union[int, None] = None) -> Callable[[float], float]:
    """
    Learning rate scheduler that is configured.
    
    Args:
        initial_value: The initial learning rate
        lr_type: The scheduler type
        total_timesteps: The total timesteps to train the agent
    Returns: 
        (function): the scheduler function
    """
    lr_type = lr_type
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    if lr_type == "cosine":
        if isinstance(final_value, str):
            final_value = float(final_value)
        iters = np.arange(total_timesteps)
        schedule = final_value + 0.5 * (initial_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        print(f"Learning Rate Scheduler : cosine, initial value : {schedule[0]}, final value : {schedule[-1]}")

        def func(progress_remaining: float) -> float:
            """
            The new learning rate
            Args:
                progress_remaining: The progress remaining - will decrease from 1 (beginning) to 0
            Returns:
                (float)
            """
            idx = int((1 - progress_remaining) *  (total_timesteps - 1))
            return schedule[idx]
    else:
        def func(progress_remaining: float) -> float:
            """
            The new learning rate
            Args:
                progress_remaining: The progress remaining - will decrease from 1 (beginning) to 0
            Returns:
                (float)
            """
            return np.max(progress_remaining * initial_value, int(1e-5))

    return func

def read_hyperparameters(algorithm: str, verbose=0, custom_hyperparams=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the default hyperparameter config for the given algorithm from
    a common YAML file. These can be overriden using the custom_hyperparams argument.

    Args:
        algorithm: The algorithm for which to get the hyperparameters
        verbose: Whether to print the final hyperparameters in the console or not
        custom_hyperparams: The hyperparameters to change/add
    """
    # Load hyperparameters from yaml file
    file_path = os.path.join(currentdir, "hyperparams.yml")
    with open(file_path) as f:
        hyperparams_dict = yaml.safe_load(f)
        # Find the correct hyperparameters based on the keys
        if algorithm in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[algorithm]
        else:
            raise ValueError(f"Hyperparameters not found for {algorithm}")

    if custom_hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(custom_hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    if verbose > 0:
        print("Default hyperparameters for environment (ones being tuned will be overridden):")
        print(saved_hyperparams)

    return hyperparams, saved_hyperparams

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        self._best_reward = 0

        self.max_hip_torque = float("-inf")
        self.min_hip_torque = float("inf")
        self.max_leg_torque = float("-inf")
        self.min_leg_torque = float("inf")


        self.max_hip_velocity = float("-inf")
        self.min_hip_velocity = float("inf")
        self.max_leg_velocity = float("-inf")
        self.min_leg_velocity = float("inf")

        # self.max_foot_pos = float("-inf")
        # self.min_foot_pos = float("inf")

        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 100  # log every 100 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        local_env = self.training_env.venv.envs[0]

        self._set_max_min_veocities(local_env)
        self._set_max_min_torques(local_env)
        #self._set_max_min_foot_positions(local_env)

        # Find the best reward
        reward = np.max(np.array(local_env.episode_returns)) if len(local_env.episode_returns) > 0 else 0
        self._best_reward = self._best_reward if self._best_reward > reward else reward


        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar("x_position", local_env.robot.GetBasePosition()[0], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("y_position", local_env.robot.GetBasePosition()[1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("z_position", local_env.robot.GetBasePosition()[2], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("x_velocity", local_env.robot.GetBaseVelocity()[0], self.num_timesteps)

            self.tb_formatter.writer.add_scalar("max_hip_torque", self.max_hip_torque, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("min_hip_torque", self.min_hip_torque, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("max_leg_torque", self.max_leg_torque, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("min_leg_torque", self.min_leg_torque, self.num_timesteps)

            self.tb_formatter.writer.add_scalar("max_hip_velocity", self.max_hip_velocity, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("min_hip_velocity", self.min_hip_velocity, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("max_leg_velocity", self.max_leg_velocity, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("min_leg_velocity", self.min_leg_velocity, self.num_timesteps)
            
            # self.tb_formatter.writer.add_scalar("hip_front_right_torque", local_env.robot.GetMotorTorques()[0], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_front_right_torque", local_env.robot.GetMotorTorques()[1], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_front_right_torque", local_env.robot.GetMotorTorques()[2], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("hip_front_left_torque", local_env.robot.GetMotorTorques()[3], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_front_left_torque", local_env.robot.GetMotorTorques()[4], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_front_left_torque", local_env.robot.GetMotorTorques()[5], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("hip_rear_right_torque", local_env.robot.GetMotorTorques()[6], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_rear_right_torque", local_env.robot.GetMotorTorques()[7], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_rear_right_torque", local_env.robot.GetMotorTorques()[8], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("hip_rear_left_torque", local_env.robot.GetMotorTorques()[9], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_rear_left_torque", local_env.robot.GetMotorTorques()[10], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_rear_left_torque", local_env.robot.GetMotorTorques()[11], self.num_timesteps)


            # self.tb_formatter.writer.add_scalar("hip_front_right_velocity", local_env.robot.GetMotorVelocities()[0], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_front_right_velocity", local_env.robot.GetMotorVelocities()[1], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_front_right_velocity", local_env.robot.GetMotorVelocities()[2], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("hip_front_left_velocity", local_env.robot.GetMotorVelocities()[3], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_front_left_velocity", local_env.robot.GetMotorVelocities()[4], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_front_left_velocity", local_env.robot.GetMotorVelocities()[5], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("hip_rear_right_velocity", local_env.robot.GetMotorVelocities()[6], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_rear_right_velocity", local_env.robot.GetMotorVelocities()[7], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_rear_right_velocity", local_env.robot.GetMotorVelocities()[8], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("hip_rear_left_velocity", local_env.robot.GetMotorVelocities()[9], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("upper_rear_left_velocity", local_env.robot.GetMotorVelocities()[10], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("lower_rear_left_velocity", local_env.robot.GetMotorVelocities()[11], self.num_timesteps)

            # foot_pos = local_env.robot.GetFootPositionsInBaseFrame().ravel()
            # self.tb_formatter.writer.add_scalar("front_right_foot_x", foot_pos[0], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("front_right_foot_y", foot_pos[1], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("front_right_foot_z", foot_pos[2], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("front_left_foot_x", foot_pos[3], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("front_left_foot_y", foot_pos[4], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("front_left_foot_z", foot_pos[5], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("rear_right_foot_x", foot_pos[6], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("rear_right_foot_y", foot_pos[7], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("rear_right_foot_z", foot_pos[8], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("rear_left_foot_x", foot_pos[9], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("rear_left_foot_y", foot_pos[10], self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("rear_left_foot_z", foot_pos[11], self.num_timesteps)

            # self.tb_formatter.writer.add_scalar("max_foot_pos", self.max_foot_pos, self.num_timesteps)
            # self.tb_formatter.writer.add_scalar("min_foot_pos", self.min_foot_pos, self.num_timesteps)

            self.tb_formatter.writer.add_scalar("best_reward", self._best_reward, self.num_timesteps)
            self.tb_formatter.writer.flush()

    def _set_max_min_torques(self, local_env):
        joint_torques = local_env.robot.GetMotorTorques()
        for i, torque in enumerate(joint_torques):
            if i % 3 == 0:
                if torque > self.max_hip_torque:
                    self.max_hip_torque = torque
                if torque < self.min_hip_torque:
                    self.min_hip_torque = torque
            else:
                if torque > self.max_leg_torque:
                    self.max_leg_torque = torque
                if torque < self.min_leg_torque:
                    self.min_leg_torque = torque
    
    def _set_max_min_veocities(self, local_env):
        joint_veocities = local_env.robot.GetMotorVelocities()
        for i, velocity in enumerate(joint_veocities):
            if i % 3 == 0:
                if velocity > self.max_hip_velocity:
                    self.max_hip_velocity = velocity
                if velocity < self.min_hip_velocity:
                    self.min_hip_velocity = velocity
            else:
                if velocity > self.max_leg_velocity:
                    self.max_leg_velocity = velocity
                if velocity < self.min_leg_velocity:
                    self.min_leg_velocity = velocity
    
    def _set_max_min_foot_positions(self, local_env):
        foot_pos = local_env.robot.GetFootPositionsInBaseFrame().ravel()
        for i, pos in enumerate(foot_pos):
                if pos > self.max_foot_pos:
                    self.max_foot_pos = pos
                if pos < self.min_foot_pos:
                    self.min_foot_pos = pos

if  __name__ == "__main__":
    scheduler = lr_schedule(3e-4, "cosine", 3e-5, 50000)
    print(scheduler(1))
    print(scheduler(0))

