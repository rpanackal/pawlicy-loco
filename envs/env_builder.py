from locomotion.envs import locomotion_gym_config
from locomotion.robots import robot_config
from locomotion.envs.sensors import robot_sensors
from locomotion.robots import a1
from locomotion.envs import locomotion_gym_env
from locomotion.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from locomotion.envs.env_wrappers import trajectory_generator_wrapper_env, simple_openloop

from tasks import walk_along_x, default_task, walk_along_x_v3, walk_along_x_v4, walk_along_x_v5, walk_along_x_v2, walk_along_x_v6
from robots import a1_v2
from sensors import a1_sensors
import numpy as np

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}

def build_regular_env(args, 
                    enable_rendering=False,
                    wrap_trajectory_generator=True,
                    action_limit=(0.75, 0.75, 0.75)
                    ):
    """ Builds the gym environment needed for RL

    Args:
        randomise_terrain: Whether to randomize terrain or not
        motor_control_mode: Position, Torque or Hybrid
        enable_rendering: Whether to configure pybullet in GUI mode or DIRECT mode
        robot_on_rack: Whether robot is on rack or not
    """

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = MOTOR_CONTROL_MODE_MAP[args.motor_control_mode]
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 10
    sim_params.enable_action_interpolation = True
    sim_params.enable_action_filter = True
    sim_params.enable_clip_motor_commands = True
    sim_params.robot_on_rack = False
    #sim_params.randomise_terrain = args.randomise_terrain

    
    gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)

    sensors = [
        robot_sensors.BasePositionSensor(dtype=np.float32),
        robot_sensors.IMUSensor(dtype=np.float32, channels=['R', 'P', 'Y', 'dR', 'dP', 'dY']),
        #robot_sensors.PoseSensor(dtype=np.float32),
        a1_sensors.FootPositionSensor(dtype=np.float32), 
        #robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS, dtype=np.float32),
        a1_sensors.MotorVelocitySensor(dtype=np.float32),
        a1_sensors.MotorTorqueSensor(dtype=np.float32)
    ]

    task = walk_along_x_v6.WalkAlongX()


    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            robot_class=a1_v2.A1V2,
                                            robot_sensors=sensors,
                                            task=task)

    env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(
        env)
    
    if (MOTOR_CONTROL_MODE_MAP[args.motor_control_mode] == robot_config.MotorControlMode.POSITION) \
        and wrap_trajectory_generator:
        env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
            env,
            trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
                action_limit=action_limit))

    return env