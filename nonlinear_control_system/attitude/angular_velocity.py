from __future__ import annotations
import numpy as np

from . import YawPitchRoll

class AngularVelocity(np.ndarray):
    def __new__(cls, input_array: np.ndarray) -> AngularVelocity:
        obj = np.asarray(input_array, dtype=float).reshape(3,1).view(cls)
        return obj
    
    def __array_finalize__(self, obj: np.ndarray) -> AngularVelocity:
        if obj is None: return
    
    @property
    def x_component(self) -> float:
        return self[0, 0]
    
    @property
    def y_component(self) -> float:
        return self[1, 0]

    @property
    def z_component(self) -> float:
        return self[2, 0]
    
    @staticmethod
    def to_ypr_rates(angular_rates: np.ndarray, attitude: YawPitchRoll, n: float) -> np.ndarray:
        """
        Convert the angular velocity to yaw, pitch, and roll rates.
        """
        assert isinstance(attitude, YawPitchRoll), "Attitude must be a YawPitchRoll object"

        matrix = AngularVelocity._calculate_ypr_rate_matrix(attitude)
        affine_vector = AngularVelocity._calculate_ypr_rate_vector(attitude, n)
        return matrix @ angular_rates + affine_vector
    
    @staticmethod
    def _calculate_ypr_rate_matrix(attitude: YawPitchRoll) -> np.matrix:
        matrix = np.array([
            [1, np.sin(attitude.roll) * np.tan(attitude.pitch), np.cos(attitude.roll) * np.tan(attitude.pitch)],
            [0, np.cos(attitude.roll), -np.sin(attitude.roll)],
            [0, np.sin(attitude.roll) / np.cos(attitude.pitch), np.cos(attitude.roll) / np.cos(attitude.pitch)]
        ])
        return matrix
    
    @staticmethod
    def _calculate_ypr_rate_vector(attitude: YawPitchRoll, n: float) -> np.ndarray:
        affine_vector = n * np.array([
            [np.sin(attitude.yaw) / np.cos(attitude.pitch)],
            [np.cos(attitude.yaw)],
            [np.tan(attitude.pitch) * np.sin(attitude.yaw)]
        ])
        return affine_vector

    
def calculate_ypr_rate_derivative(attitude: YawPitchRoll, angular_velocity: np.ndarray, n: float) -> np.ndarray:
    """
    Calculate the derivative of the yaw, pitch, and roll rates.
    """
    assert isinstance(attitude, YawPitchRoll), "Attitude must be a YawPitchRoll object"
    assert angular_velocity.size == 3, "Angular velocity must have 3 components"
    roll, pitch, yaw = attitude.roll, attitude.pitch, attitude.yaw
    omega_1, omega_2, omega_3 = angular_velocity.flatten()

    a11 = (np.cos(roll) * omega_2 - np.sin(roll) * omega_3) * np.tan(pitch)
    a12 = (np.sin(roll) * omega_2 + np.cos(roll) * omega_3) / np.cos(pitch)**2 + n * np.sin(yaw) * np.tan(pitch) / np.cos(pitch)
    a13 = -n / np.cos(pitch) * np.cos(yaw)
    a21 = -np.sin(roll) * omega_2 - np.cos(roll) * omega_3
    a22 = 0
    a23 = -n * np.sin(yaw)
    a31 = (np.cos(roll) * omega_2 - np.sin(roll) * omega_3) / np.cos(pitch)
    a32 = (np.sin(roll) * omega_2 + np.cos(roll) * omega_3) * np.tan(pitch) / np.cos(pitch) + n * np.sin(yaw) / np.cos(pitch)**2
    a33 = n * np.tan(pitch) * np.cos(yaw)
    
    return np.array([
        [a11, a12, a13, 1, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)],
        [a21, a22, a23, 0, np.cos(roll), -np.sin(roll)],
        [a31, a32, a33, 0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch)]
    ])

def ypr_rates_to_angular_velocity(ypr_rates: np.ndarray, attitude: YawPitchRoll, n: float) -> AngularVelocity:
    """ Convert yaw, pitch, and roll rates to angular velocities about body fixed axes. """
    assert isinstance(attitude, YawPitchRoll), "Attitude must be a YawPitchRoll object"
    raise NotImplementedError("This function is not implemented yet. It requires the inverse of the transformation matrix used in to_ypr_rates.")
