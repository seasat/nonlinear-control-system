from __future__ import annotations
import numpy as np

from attitude import Attitude
import dynamics
import integrator


class StateVector:
    def __init__(self, attitude: Attitude, angular_rates: np.ndarray) -> None:
        assert isinstance(attitude, Attitude)
        assert angular_rates.shape == (3, 1), "angular rates must be a 3x1 array"

        self.attitude = attitude
        self.AttitudeType = type(attitude)
        self.angular_rates = angular_rates
    
    def to_vector(self) -> np.ndarray:
        return np.vstack([self.attitude.to_vector(), self.angular_rates])
    
    def integrate_state(self, time_step: float, inertia_tensor: np.ndarray, torque: np.ndarray, mean_motion: float) -> tuple[Attitude, np.ndarray]:
        """ Integrate the state vector using the Runge-Kutta method.  """
        state_vector: np.ndarray = self.to_vector()
        new_state_vector: np.ndarray = integrator.rk4(
            self.calculate_state_derivative,
            state_vector,
            0,
            time_step,
            inertia_tensor,
            torque,
            mean_motion
        )
        attitude_coefficients: np.ndarray = new_state_vector[0:len(self.attitude.to_vector())]
        attitude: Attitude = self.AttitudeType(attitude_coefficients)
        angular_rates: np.ndarray = new_state_vector[len(self.attitude.to_vector()):]
        return attitude, angular_rates

    @staticmethod
    def calculate_state_derivative(state: StateVector, time: None, inertia_tensor: np.ndarray, torque: np.ndarray, mean_motion: float) -> np.ndarray:
        attitude_rates = state.attitude.calculate_derivative(state.angular_rates, mean_motion)
        angular_accelerations = dynamics.calculate_angular_acceleration(state.angular_rates, inertia_tensor, torque, mean_motion)
        return np.vstack([attitude_rates, angular_accelerations])
