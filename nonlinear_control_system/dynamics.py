import numpy as np
import control

from attitude import YawPitchRoll, AngularVelocity


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
    ypr_rates = AngularVelocity.to_ypr_rates(angular_velocity, attitude, mean_motion)

    state_change = np.concatenate((ypr_rates, angular_acceleration))
    return state_change

def get_linearized_system(inertia_tensor: np.matrix, mean_motion: float) -> control.StateSpace:
    """
    Get the system matrices A and B for the linearized system
    dy/dt = A @ y + B @ u
    """
    J_INV = np.linalg.inv(inertia_tensor)

    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)  # Identity matrix for angular velocity
    a[0, 1] = mean_motion
    orbital_coupling = np.array([
        [0, 0, -mean_motion * (inertia_tensor[2,2] + inertia_tensor[1,1])],
        [0, 0, 0],
        [mean_motion * (inertia_tensor[0,0] + inertia_tensor[1,1]), 0, 0],
    ])
    a[3:6, 3:6] = -np.linalg.inv(inertia_tensor) @ orbital_coupling

    b = np.zeros((6, 3))
    # control torque affect the angular velocity
    b[3:6, 0:3] = J_INV

    # track all outputs
    c = np.eye(6)
    # no feedthrough
    d = np.zeros((6, 3))

    return control.StateSpace(a, b, c, d)
