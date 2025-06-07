from __future__ import annotations
from enum import Enum
import numpy as np

from . import Attitude
from . import DirectionCosineMatrix as dcm
from . import Quaternion


class Axis(Enum):
    """
    Enum for the axes of rotation.
    """
    X_ROTATION = dcm.calculate_t_1
    Y_ROTATION = dcm.calculate_t_2
    Z_ROTATION = dcm.calculate_t_3


class EulerAngles(Attitude):
    vector_length: int = 3  # Length of the vector representation of the Euler angles

    def __init__(self,
        first_axis: Axis, first_angle: float,
        second_axis: Axis, second_angle: float,
        third_axis: Axis, third_angle: float
    ) -> None:
        """
        Initialize the EulerAngles class with three angles and their corresponding axes.
        """
        super().__init__()
        # no two successive axes can be the same
        assert first_axis != second_axis and second_axis != third_axis, "No two successive axes can be the same"

        self.first_axis = first_axis
        self.first_angle = first_angle
        self.second_axis = second_axis
        self.second_angle = second_angle
        self.third_axis = third_axis
        self.third_angle = third_angle
    
    def to_dcm(self) -> dcm:
        """
        Convert the Euler angles to a direction cosine matrix.
        """
        t_1 = self.first_axis(self.first_angle)
        t_2 = self.second_axis(self.second_angle)
        t_3 = self.third_axis(self.third_angle)
        return t_3 @ t_2 @ t_1
    
    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to be between -pi and pi."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    

class YawPitchRoll(EulerAngles):
    """
    Yaw-Pitch-Roll (Z-Y-X) Euler angles.
    """
    def __init__(self, angles: np.ndarray) -> None:
        """
        Initialize YawPitchRoll with vector of roll then pitch then yaw angle.
        Note that rotations are performed in reverse order yaw -> pitch -> roll.
        """
        flattened = np.asarray(angles).flatten()
        super().__init__(
            Axis.Z_ROTATION, self.wrap_angle(flattened[2]),
            Axis.Y_ROTATION, self.wrap_angle(flattened[1]),
            Axis.X_ROTATION, self.wrap_angle(flattened[0])
        )
    
    @property
    def yaw(self) -> float:
        return self.first_angle
    @property
    def pitch(self) -> float:
        return self.second_angle
    @property
    def roll(self) -> float:
        return self.third_angle
    
    def to_vector(self) -> np.ndarray:
        """Return roll, pitch and yaw as a column vector."""
        return np.array([self.roll, self.pitch, self.yaw]).reshape(3, 1)

    def calculate_derivative(self, body_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        assert body_rates.shape == (3, 1), "Body rates must be a 3x1 ndarray"
        matrix = self._calculate_ypr_rate_matrix()
        affine_vector = self._calculate_ypr_rate_vector(mean_motion)
        return matrix @ body_rates + affine_vector

    def _calculate_ypr_rate_matrix(self) -> np.ndarray:
        matrix = np.array([
            [1, np.sin(self.roll) * np.tan(self.pitch), np.cos(self.roll) * np.tan(self.pitch)],
            [0, np.cos(self.roll), -np.sin(self.roll)],
            [0, np.sin(self.roll) / np.cos(self.pitch), np.cos(self.roll) / np.cos(self.pitch)]
        ])
        return matrix
    
    def _calculate_ypr_rate_vector(self, mean_motion: float) -> np.ndarray:
        affine_vector = mean_motion * np.array([
            [np.sin(self.yaw) / np.cos(self.pitch)],
            [np.cos(self.yaw)],
            [np.tan(self.pitch) * np.sin(self.yaw)]
        ])
        return affine_vector

    def calculate_derivative_state_derivative(self, body_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        """
        Calculate the derivative of the yaw, pitch, and roll rates.
        """
        assert body_rates.shape == (3, 1), "Body rates must be a 3x1 ndarray"
        roll, pitch, yaw = self.roll, self.pitch, self.yaw
        omega_1, omega_2, omega_3 = body_rates[0, 0], body_rates[1, 0], body_rates[2, 0]
        n = mean_motion

        a11 = (np.cos(roll) * omega_2 - np.sin(roll) * omega_3) * np.tan(pitch)
        a12 = (np.sin(roll) * omega_2 + np.cos(roll) * omega_3) / np.cos(pitch)**2 + n * np.sin(yaw) * np.tan(pitch) / np.cos(pitch)
        a13 = -n / np.cos(pitch) * np.cos(yaw)
        a21 = -np.sin(roll) * omega_2 - np.cos(roll) * omega_3
        a22 = 0
        a23 = -n * np.sin(yaw)
        a31 = (np.cos(roll) * omega_2 - np.sin(roll) * omega_3) / np.cos(pitch)
        a32 = (np.sin(roll) * omega_2 + np.cos(roll) * omega_3) * np.tan(pitch) / np.cos(pitch) + n * np.sin(yaw) / np.cos(pitch)**2
        a33 = n * np.tan(pitch) * np.cos(yaw)
        a = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])

        return np.hstack([a, self._calculate_ypr_rate_matrix()])

    def derivative_to_body_rates(self, attitude_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        assert attitude_rates.shape == (3, 1), "Attitude rates must be a 3x1 ndarray"
        
        matrix = self._calculate_ypr_rate_matrix()
        affine_vector = self._calculate_ypr_rate_vector(mean_motion)
        return np.linalg.inv(matrix) @ (self - affine_vector)

    def __add__(self, other: YawPitchRoll) -> YawPitchRoll:
        """
        Add two attitudes in yaw-pitch-roll representation together.
        Wrap the angles to be between -pi and pi.
        """
        assert isinstance(other, YawPitchRoll), "Can only add YawPitchRoll objects"
        new_angles = [self.wrap_angle(angle) for angle in [self.roll + other.roll, self.pitch + other.pitch, self.yaw + other.yaw]]
        return YawPitchRoll(new_angles)
    
    def get_error(self, other: YawPitchRoll) -> YawPitchRoll:
        return self - other
    
    def __sub__(self, other: YawPitchRoll) -> YawPitchRoll:
        """
        Subtract two attitudes in yaw-pitch-roll representation.
        Wrap the angles to be between -pi and pi.
        """
        assert isinstance(other, YawPitchRoll), "Can only subtract YawPitchRoll objects"
        new_angles = [self.wrap_angle(angle) for angle in [self.roll - other.roll, self.pitch - other.pitch, self.yaw - other.yaw]]
        return YawPitchRoll(new_angles)

    def to_quaternion(self) -> Quaternion:
        """ Convert the yaw-pitch-roll angles to a quaternion.  """
        roll, pitch, yaw = self.roll, self.pitch, self.yaw

        q1 = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        q2 = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        q3 = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        q4 = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return Quaternion([q1, q2, q3, q4])
