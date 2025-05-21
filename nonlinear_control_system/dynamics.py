import numpy as np

from attitude import YawPitchRoll, AngularVelocity


def calculate_angular_acceleration(angular_velocity: np.ndarray, inertia_tensor: np.matrix, torque: np.ndarray) -> np.ndarray:
    angular_acceleration = np.linalg.inv(inertia_tensor) @ (torque - np.cross(
        angular_velocity.flatten(), (inertia_tensor @ angular_velocity).flatten()
    ).reshape(3, 1))
    return angular_acceleration

def calculate_state_change(state: np.ndarray, time: None, inertia_tensor: np.ndarray, torque: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = state[:3]
    attitude = YawPitchRoll([roll, pitch, yaw])
    angular_velocity = state[3:6]

    angular_acceleration = calculate_angular_acceleration(angular_velocity, inertia_tensor, torque)
    ypr_rates = AngularVelocity.to_ypr_rates(angular_velocity, attitude)

    state_change = np.concatenate((ypr_rates, angular_acceleration))
    return state_change
