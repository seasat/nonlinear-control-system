from collections.abc import Callable
import numpy as np
from attitude import Attitude
import dynamics

def rk4(attitude: Attitude, angular_rates: np.ndarray, time_step: float, inertia_tensor, torque, mean_motion, time=0):
    """Runge-Kutta 4th order integration method."""
    state_vector = np.vstack([attitude.to_vector(), angular_rates])

    k1 = derivative(state_vector, time, attitude, inertia_tensor, torque, mean_motion)
    k2 = derivative(state_vector + 0.5 * time_step * k1, time + 0.5 * time_step, attitude, inertia_tensor, torque, mean_motion)
    k3 = derivative(state_vector + 0.5 * time_step * k2, time + 0.5 * time_step, attitude, inertia_tensor, torque, mean_motion)
    k4 = derivative(state_vector + time_step * k3, time + time_step, attitude, inertia_tensor, torque, mean_motion)
    
    new_state_vector = state_vector + (time_step / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return decompose_state_vector(new_state_vector, type(attitude))
    

def derivative(state_vector: np.ndarray, time: float, old_attitude: Attitude, inertia_tensor: np.ndarray, torque: np.ndarray, mean_motion: float) -> np.ndarray:
    """ Calculate the derivative of the state vector. """
    attitude, angular_rates = decompose_state_vector(state_vector, type(old_attitude))

    attitude_rates: np.ndarray = attitude.calculate_derivative(angular_rates, mean_motion)
    angular_accelerations: np.ndarray = dynamics.calculate_angular_acceleration(angular_rates, inertia_tensor, torque, mean_motion)
    
    return np.vstack([attitude_rates, angular_accelerations])

def decompose_state_vector(state_vector: np.ndarray, attitude_type: type[Attitude]) -> tuple[Attitude, np.ndarray]:
    """ Decompose a state vector into an attitude and angular rates. """
    attitude_length: int = attitude_type.vector_length
    attitude_components: np.ndarray = state_vector[:attitude_length]
    angular_rates: np.ndarray = state_vector[attitude_length:]
    
    return attitude_type(attitude_components), angular_rates
