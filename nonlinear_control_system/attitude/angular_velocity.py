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
    def to_ypr_rates(angular_rates: np.ndarray, attitude: YawPitchRoll) -> np.ndarray:
        """
        Convert the angular velocity to yaw, pitch, and roll rates.
        """
        assert isinstance(attitude, YawPitchRoll), "Attitude must be a YawPitchRoll object"
        matrix = np.array([
            [1, np.sin(attitude.yaw) * np.tan(attitude.pitch), np.cos(attitude.yaw) * np.tan(attitude.pitch)],
            [0, np.cos(attitude.yaw), -np.sin(attitude.yaw)],
            [0, np.sin(attitude.yaw) / np.cos(attitude.pitch), np.cos(attitude.yaw) / np.cos(attitude.pitch)]
        ])
        return np.asarray(matrix @ angular_rates).reshape(3, 1)
