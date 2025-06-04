from __future__ import annotations
import numpy as np


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


#class YPRRates(AngularVelocity):
    #def to_body_rates(self, attitude: YawPitchRoll, n: float) -> AngularVelocity:
        #""" Convert yaw, pitch, and roll rates to angular velocities about body fixed axes. """
        #assert isinstance(attitude, YawPitchRoll), "Attitude must be a YawPitchRoll object"
        
        #matrix = YPRRates._calculate_ypr_rate_matrix(attitude)
        #affine_vector = YPRRates._calculate_ypr_rate_vector(attitude, n)
        #return np.linalg.inv(matrix) @ (self - affine_vector)
    