# from pawlicy.envs import A1GymEnv
# from pawlicy.envs.gym_config import LocomotionGymConfig
# from pawlicy.robots import robot_config
# from pawlicy.robots.a1 import constants
# from pawlicy.sensors import robot_sensors
# from pawlicy.tasks import walk_along_x
from learning import trainer, utils

import os
import inspect
import argparse

from envs.a1_gym_env import A1GymEnv
#from envs.env_builder import build_regular_env

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")

def parse_arguements():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', "-m", dest="mode", default="test", choices=["train", "test", "random"], type=str, help='to set to training or testing mode')
    parser.add_argument('--max_episode_steps', "-mes", dest="max_episode_steps", default=1000, type=int, help='maximum steps per episode')
    parser.add_argument('--visualize', "-v", dest="visualize", action="store_true", help='To flip rendering behaviour')
    parser.add_argument("--randomise_terrain", "-rt", dest="randomise_terrain", default=False, type=bool, help="to setup a randommized terrain")
    parser.add_argument("--motor_control_mode", "-mcm", dest="motor_control_mode",  default="position", choices=["position", "torque", "hybrid"], type=str, help="to set motor control mode")

    parser.add_argument('--author', "-au", dest="author", default="rpanackal", type=str, help='name of author')
    parser.add_argument('--exp_suffix', "-s", dest="exp_suffix", default="", type=str, help='appends to experiment name')
    parser.add_argument('--total_timesteps', "-tts", dest="total_timesteps", default=int(1e5), type=int, help='total number of training steps')
    
    parser.add_argument('--total_num_eps', "-tne", dest="total_num_eps", default=20, type=int, help='total number of test episodes')
    parser.add_argument('--load_exp_name', "-l", dest="load_exp_name", default="sac_rpanackal_tns100000", type=str, help='name of experiment to be tested')
    #parser.add_argument('--mode', "-m", default="eval", choices=["train", "eval"], type=str, help='To set to training or evaluation mode')

    args = parser.parse_args()
    
    args.motor_control_mode = args.motor_control_mode.capitalize()
    return args

def main():

    args = parse_arguements()

    # Training
    if args.mode == "train":
        env = A1GymEnv(args, enable_rendering=False)

        # Train the agent
        local_trainer = trainer.Trainer(env, "SAC", args)
        _, hyperparameters = utils.read_hyperparameters("SAC", 1, {"learning_starts": 2000})
        model = local_trainer.train(hyperparameters)

        # Save the model after training
        local_trainer.save_model(SAVE_DIR)

    # Testing
    elif args.mode == "test":
        test_env = A1GymEnv(args, enable_rendering=True)
        trainer.Trainer(test_env, "SAC", args).test(SAVE_DIR)
    
    elif args.mode == "random":
        rand_env = A1GymEnv(args, enable_rendering=True)
        trainer.Trainer(rand_env, "SAC", args).random()
 
if __name__ == "__main__":
    main()
