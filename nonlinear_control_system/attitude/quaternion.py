from __future__ import annotations
import numpy as np

from . import Attitude#, DirectionCosineMatrix, ClassicalRodriguezParameter, ModifiedRodriguezParameter


class Quaternion(Attitude):
    def __init__(self, q1: float, q2: float, q3: float, q4: float) -> None:
        """
        Initialize the Quaternion class with a quaternion.

        :param q1: The first component of the quaternion.
        :param q2: The second component of the quaternion.
        :param q3: The third component of the quaternion.   
        :param q4: The fourth component of the quaternion.
        """
        super().__init__()
        assert np.isclose(np.linalg.norm([q1, q2, q3, q4]), 1), "Quaternion must be a unit quaternion"
        
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
    
    def to_vector(self) -> np.ndarray:
        """ Return as column vector. """
        return np.array([[self.q1], [self.q2], [self.q3], [self.q4]])
    
    def __matmul__(self, other: Quaternion) -> Quaternion:
        """
        Successive rotations of two quaternions.

        :param other: The other quaternion to multiply with.
        :return: The resulting attitude as a quaternion.
        """
        assert isinstance(other, Quaternion), "Can only multiply with another Quaternion"
        
        return Quaternion.from_successive_rotations(self, other)
    
    @classmethod
    def from_successive_rotations(cls, q1: Quaternion, q2: Quaternion) -> Quaternion:
        """
        Initialize the Quaternion class with two quaternions.

        :param q1: The first quaternion.
        :param q2: The second quaternion.
        """
        super().__init__()
        
        quaternion_matrix = np.asmatrix([
            [q2.q4, q2.q3, -q2.q2, q2.q1],
            [-q2.q3, q2.q4, q2.q1, q2.q2],
            [q2.q2, -q2.q1, q2.q4, q2.q3],
            [-q2.q1, -q2.q2, -q2.q3, q2.q4]
        ])
        result = quaternion_matrix @ np.asmatrix([q1.q1, q1.q2, q1.q3, q1.q4]).T

        q1 = result[0, 0]
        q2 = result[1, 0]
        q3 = result[2, 0]
        q4 = result[3, 0]
        return cls(q1, q2, q3, q4)
    
    ## conversion methods
    #def to_dcm(self) -> DirectionCosineMatrix:
        #"""
        #Convert the quaternion to a direction cosine matrix.

        #:return: An instance of DirectionCosineMatrix.
        #"""
        #c11 = 1 - 2 * (self.q2**2 + self.q3**2)
        #c12 = 2 * (self.q1 * self.q2 + self.q3 * self.q4)
        #c13 = 2 * (self.q1 * self.q3 - self.q2 * self.q4)
        #c21 = 2 * (self.q1 * self.q2 - self.q3 * self.q4)
        #c22 = 1 - 2 * (self.q1**2 + self.q3**2)
        #c23 = 2 * (self.q2 * self.q3 + self.q1 * self.q4)
        #c31 = 2 * (self.q1 * self.q3 + self.q2 * self.q4)
        #c32 = 2 * (self.q2 * self.q3 - self.q1 * self.q4)
        #c33 = 1 - 2 * (self.q1**2 + self.q2**2)

        #return DirectionCosineMatrix(np.asmatrix([
            #[c11, c12, c13],
            #[c21, c22, c23],
            #[c31, c32, c33]
        #]))
        
    #def to_crp(self) -> ClassicalRodriguezParameter:
        #"""
        #Convert the quaternion to a Classical Rodriguez parameter.

        #:return: An instance of ClassicalRodriguezParameter.
        #"""
        #from . import ClassicalRodriguezParameter
        #tau1 = self.q1 / self.q4
        #tau2 = self.q2 / self.q4
        #tau3 = self.q3 / self.q4
        #return ClassicalRodriguezParameter(tau1, tau2, tau3)
    
    #def to_mrp(self) -> ModifiedRodriguezParameter:
        #"""
        #Convert the quaternion to a modified Rodriguez parameter.

        #:return: An instance of ModifiedRodriguezParameter.
        #"""
        #from . import ModifiedRodriguezParameter
        #sigma1 = self.q1 / (1 + self.q4)
        #sigma2 = self.q2 / (1 + self.q4)
        #sigma3 = self.q3 / (1 + self.q4)
        #return ModifiedRodriguezParameter(sigma1, sigma2, sigma3)
