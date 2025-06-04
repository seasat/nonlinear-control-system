from __future__ import annotations
from enum import Enum
import numpy as np

from . import Attitude
from . import DirectionCosineMatrix as dcm
from . import BodyRates


class Axis(Enum):
    """
    Enum for the axes of rotation.
    """
    X_ROTATION = dcm.calculate_t_1
    Y_ROTATION = dcm.calculate_t_2
    Z_ROTATION = dcm.calculate_t_3


class EulerAngles(Attitude):
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

    def calculate_derivative(self, body_rates: BodyRates, mean_motion: float) -> np.ndarray:
        return body_rates.to_ypr_rates(self, mean_motion)

    def __add__(self, other: YawPitchRoll) -> YawPitchRoll:
        """
        Add two attitudes in yaw-pitch-roll representation together.
        Wrap the angles to be between -pi and pi.
        """
        assert isinstance(other, YawPitchRoll), "Can only add YawPitchRoll objects"
        new_angles = [self.wrap_angle(angle) for angle in [self.roll + other.roll, self.pitch + other.pitch, self.yaw + other.yaw]]
        return YawPitchRoll(new_angles)
    
    def __sub__(self, other: YawPitchRoll) -> YawPitchRoll:
        """
        Subtract two attitudes in yaw-pitch-roll representation.
        Wrap the angles to be between -pi and pi.
        """
        assert isinstance(other, YawPitchRoll), "Can only subtract YawPitchRoll objects"
        new_angles = [self.wrap_angle(angle) for angle in [self.roll - other.roll, self.pitch - other.pitch, self.yaw - other.yaw]]
        return YawPitchRoll(new_angles)
