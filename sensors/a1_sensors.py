from locomotion.envs.sensors import sensor

import numpy as np
import typing

_ARRAY = typing.Iterable[float] #pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] #pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any] #pylint: disable=invalid-name

NUM_MOTORS = 12
NUM_LEGS = 4
JOINT_MAX_VELOCITIES = [52.4, 28.6, 28.6] * NUM_LEGS
JOINT_MAX_FORCES = [20, 55, 55] * NUM_LEGS

OBSERVED_VELOCITY_LOW = [-100] * NUM_MOTORS
OBSERVED_VELOCITY_HIGH = [100] * NUM_MOTORS

OBSERVED_TORQUE_LOW = [-50, -700, -700] * NUM_LEGS
OBSERVED_TORQUE_HIGH = [50, 300, 300] * NUM_LEGS
#JOINT_MAX_TORQUE = 50

class MotorVelocitySensor(sensor.BoxSpaceSensor):
    """A Sensor that reads motor velocities"""    

    def __init__(self,
                num_motors: int = NUM_MOTORS,
                noisy_reading: bool = True, 
                name: typing.Text = "MotorVelocities", 
                lower_bound: _FLOAT_OR_ARRAY = OBSERVED_VELOCITY_LOW,
                upper_bound: _FLOAT_OR_ARRAY = OBSERVED_VELOCITY_HIGH, 
                dtype: typing.Type[typing.Any] = np.float64) -> None:

        self._num_motors = num_motors
        self._noisy_reading = noisy_reading

        super(MotorVelocitySensor, self).__init__(
            name=name, 
            shape=(num_motors,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
    
    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_velocities = self._robot.GetMotorVelocities()
        else:
            motor_velocities = self._robot.GetTrueMotorVelocities()

        return motor_velocities
    
class MotorTorqueSensor(sensor.BoxSpaceSensor):
    def __init__(self,
            num_motors: int = NUM_MOTORS,
            noisy_reading: bool = True, 
            name: typing.Text = "MotorTorques", 
            lower_bound: _FLOAT_OR_ARRAY = OBSERVED_TORQUE_LOW,
            upper_bound: _FLOAT_OR_ARRAY = OBSERVED_TORQUE_HIGH, 
            dtype: typing.Type[typing.Any] = np.float64) -> None:

        self._num_motors = num_motors
        self._noisy_reading = noisy_reading

        super(MotorTorqueSensor, self).__init__(
            name=name, 
            shape=(num_motors,), 
            lower_bound=lower_bound, 
            upper_bound=upper_bound, 
            dtype=dtype)
    
    def _get_observation(self) -> _ARRAY:
        if self._noisy_reading:
            motor_torques = self._robot.GetMotorTorques()
        else:
            motor_torques = self._robot.GetTrueMotorTorques()

        return motor_torques

