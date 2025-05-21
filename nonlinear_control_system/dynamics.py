import numpy as np


def calculate_angular_acceleration(
    angular_velocity: np.ndarray,
    time: None,
    inertia_tensor: np.matrix, 
    torque: np.ndarray,
) -> np.ndarray:
    angular_acceleration = np.linalg.inv(inertia_tensor) @ (torque - np.cross(
        angular_velocity.flatten(), (inertia_tensor @ angular_velocity).flatten()
    ).reshape(3, 1))
    return angular_acceleration