import os
import datetime
from gym.wrappers import TimeLimit
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import numpy as np

# TODO: Check the logic of NormalizeActionWrapper
from envs.wrappers.normalize_actions_wrapper import NormalizeActionWrapper
from learning import utils

ALGORITHMS = {"SAC": SAC, "PPO": PPO, "TD3": TD3}


class Trainer:
    """
    The trainer class provides some basic methods to train an agent using different algorithms
    available in stable_baselines3

    Args:
        env: The gym environment to train on.
        algorithm: The algorithm to use.
        max_episode_steps: The no. of steps per episode
    """
    def __init__(self, env, args, eval_env=None):
        self._algorithm = args.algorithm
        self._max_episode_steps = args.max_episode_steps
        self._total_timesteps = args.total_timesteps
        self._log_dir = args.log_dir

        self._args = args 
        self._env = self.setup_env(env, self._max_episode_steps)

        if eval_env is not None:
            self._eval_env = self.setup_env(eval_env, self._max_episode_steps // 2)
        else:
            self._eval_env = eval_env

        self._exp_name = self._get_exp_name()
        self._exp_dir = os.path.join(self._log_dir, self._exp_name)

    def train(self, total_timesteps=None, eval_env=None):
        """
        Trains an agent to use the environment to maximise the rewards while performing
        a specific task. This will tried out with multiple other algorithms later for
        benchmarking purposes.

        Args:
            env: The gym environment to train the agent on.
            algorithm: The algorithm to use
            hyperparameters: The hyperparameters to use
            n_timesteps: The number of timesteps to train
            eval_env: The gym environment used for evaluation.
        """
        if total_timesteps is not None:
            self._total_timesteps = total_timesteps

        override_hyperparams = {
            "n_timesteps": self._total_timesteps,
        }
        
        _, hyperparameters = utils.read_hyperparameters(self._algorithm, 1, override_hyperparams)
        # Sanity checks
        n_timesteps = hyperparameters.pop("n_timesteps", None)
        if n_timesteps is None:
            raise ValueError("The hyperparameter 'n_timesteps' is missing.")
        eval_frequency = hyperparameters.pop("eval_freq", 5000)
        scheduler_type = hyperparameters.pop("learning_rate_scheduler", None)
        lr = hyperparameters.pop("learning_rate", float(1e-3))
        noise_type = hyperparameters.pop("noise_type", "normal")
        noise_std = hyperparameters.pop("noise_std", 0.0)

        # The noise objects for TD3
        if self._algorithm == "TD3":
            policy_kwargs = dict(net_arch=[400, 300])
            if noise_type == "normal":
                action_noise = NormalActionNoise(mean=np.zeros(12), sigma=noise_std * np.ones(12))

        # Setup up learning rate scheduler arguments, if needed
        if scheduler_type is not None:
            lr_scheduler_args = {
                "lr_type": scheduler_type,
                "total_timesteps": n_timesteps,
                "final_value": 3e-6
            }

        tensorboard_log_dir = os.path.join(self._exp_dir, "logs")
         # Use the appropriate algorithm
        self._model = ALGORITHMS[self._algorithm](env=self._env,
                                                    verbose=1,
                                                    action_noise=action_noise if self._algorithm == "TD3" else None,
                                                    learning_rate=utils.lr_schedule(lr, **lr_scheduler_args) if scheduler_type is not None else lr,
                                                    tensorboard_log=tensorboard_log_dir,
                                                    **hyperparameters,
                                                    policy_kwargs=policy_kwargs if self._algorithm == "TD3" else None)


        # Train the model (check if evaluation is needed)
        
        if self._eval_env is not None:
            eval_callback = EvalCallback(self._eval_env,
                                            best_model_save_path=self._exp_dir,
                                            log_path=self._exp_dir,
                                            eval_freq=eval_frequency,
                                            deterministic=True,
                                            render=self._args.visualize)
            callbacks = CallbackList([eval_callback, utils.TensorboardCallback()])

            self._model.learn(n_timesteps,
                                log_interval=100,
                                eval_env=self._eval_env,
                                eval_freq=eval_frequency,
                                reset_num_timesteps=False,
                                callback=callbacks)
        else:
            self._model.learn(n_timesteps,
                                log_interval=100,
                                reset_num_timesteps=False,
                                callback=utils.TensorboardCallback())

        # Return the trained model
        return self._model

    def _get_exp_name(self):
        time_stamp = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        exp_name = f"{self._algorithm}_{self._args.author}_{time_stamp}_tts{self._total_timesteps}"
        if self._args.exp_suffix:
            exp_name = f"{exp_name}_{self._args.exp_suffix}"
        
        return exp_name

    def save_model(self, save_dir=None, store_replay_buffer=False):
        """
        Saves the model in trainer. Also can save the replay buffer

        Args:
            save_path: The path to save the model in
            save_replay_buffer: Whether to save the replay buffer or not
        """
        if save_dir is None:
            raise ValueError("No path specified to save the trained model.")
        else:

            # Create the directory to save the models in.
            os.makedirs(save_dir, exist_ok=True)
            self._model.save(os.path.join(self._exp_dir, "model"))
            if store_replay_buffer:
                self._model.save_replay_buffer(os.path.join(self._exp_dir, "replay_buffer"))
        
    def load_model(self, exp_name=None, model_path=None, load_replay_buffer=False):
        if model_path is not None:
            model = ALGORITHMS[self._algorithm].load(model_path)
        
        elif exp_name is not None:
            # If experiment name given then looks for model in _log_dir directory

            # Loads the best model by default, if not available then loads final model
            if self._args.load_final_model or not os.path.exists(os.path.join(self._log_dir, exp_name, "best_model.zip")):
                model_path = os.path.join(self._log_dir, exp_name, "model")
            else:
                print("Best experiment model is being loaded... ")
                model_path = os.path.join(self._log_dir, exp_name, "best_model")
            
            model = ALGORITHMS[self._algorithm].load(model_path)
            if load_replay_buffer:
                model.load_replay_buffer(os.path.join(self._log_dir, exp_name, "replay_buffer"))
        
        else:
            raise ValueError("No model path or experiment name specified to load a trained model.")
        
        return model
    
    def test(self, exp_name=None, model_path=None):
        """
        Tests the agent

        Args:
            env: The gym environment to test the agent on.
        """
        self._model = self.load_model(exp_name, model_path)

        for i in range(self._args.total_num_eps):
            done = False
            obs = self._env.reset()
            while not done:
                action, _states = self._model.predict(obs, deterministic=True)
                obs, reward, done, info = self._env.step(action)

    def random(self):
        """
        Tests the agent

        Args:
            env: The gym environment to test with random runs.
        """
        for i in range(self._args.total_num_eps):
            done = False
            obs = self._env.reset()
            while not done:
                obs, reward, done, info = self._env.step(self._env.action_space.sample())
    
    def setup_env(self, env, max_episode_steps):
        """
        Modifies the environment to suit to the needs of stable_baselines3.

        Args:
            max_episode_steps: The number of steps per episode
        """
        # Normalize the action space
        env = NormalizeActionWrapper(env)
        # Set the number of steps for each episode
        env = TimeLimit(env, max_episode_steps)
        # To monitor training stats
        env = Monitor(env)
        check_env(env)
        # a simple vectorized wrapper
        env = DummyVecEnv([lambda: env])
        # Normalizes the observation space and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        return env

    @property
    def model(self):
        return self._model

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        self._max_episode_steps = value
