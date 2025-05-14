from enum import Enum

from . import Attitude
from . import DirectionCosineMatrix as dcm


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
    

class YawPitchRoll(EulerAngles):
    """
    Yaw-Pitch-Roll (Z-Y-X) Euler angles.
    """
    def __init__(self, yaw: float, pitch: float, roll: float) -> None:
        super().__init__(
            Axis.Z_ROTATION, yaw,
            Axis.Y_ROTATION, pitch,
            Axis.X_ROTATION, roll
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
