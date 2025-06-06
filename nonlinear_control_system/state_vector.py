from __future__ import annotations
import numpy as np

from attitude import Attitude
import dynamics

class StateVector:
    def __init__(self, attitude: Attitude, angular_rates: np.ndarray) -> None:
        assert isinstance(attitude, Attitude)
        assert angular_rates.shape == (3, 1), "angular rates must be a 3x1 array"

        self.attitude = attitude
        self.angular_rates = angular_rates
    
    def to_vector(self) -> np.ndarray:
        return np.vstack([self.attitude.to_vector(), self.angular_rates])

    @staticmethod
    def calculate_state_derivative(state: StateVector, time: None, inertia_tensor: np.ndarray, torque: np.ndarray, mean_motion: float) -> np.ndarray:
        attitude_rates = state.attitude.calculate_derivative(state.angular_rates, mean_motion)
        angular_accelerations = dynamics.calculate_angular_acceleration(state.angular_rates, inertia_tensor, torque, mean_motion)
        return np.vstack([attitude_rates, angular_accelerations])
