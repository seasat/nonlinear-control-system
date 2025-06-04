import numpy as np

from attitude import YawPitchRoll


def calculate_angular_acceleration(angular_velocity: np.ndarray, inertia_tensor: np.matrix, torque: np.ndarray, mean_motion: float) -> np.ndarray:
    lvlh_velocity = np.array([[0], [-mean_motion], [0]])
    inertial_velocity = angular_velocity + lvlh_velocity
    angular_acceleration = np.linalg.inv(inertia_tensor) @ (torque - np.cross(
        inertial_velocity.flatten(),
        (inertia_tensor @ inertial_velocity).flatten()
    ).reshape(3, 1))
    return angular_acceleration

def calculate_state_change(state: np.ndarray, time: None, inertia_tensor: np.ndarray, torque: np.ndarray, mean_motion: float) -> np.ndarray:
    roll, pitch, yaw = state[:3]
    attitude = YawPitchRoll([roll, pitch, yaw])
    angular_velocity = state[3:6]

    angular_acceleration = calculate_angular_acceleration(angular_velocity, inertia_tensor, torque, mean_motion)
    ypr_rates = angular_velocity.to_ypr_rates(attitude, mean_motion)

    state_change = np.concatenate((ypr_rates, angular_acceleration))
    return state_change

