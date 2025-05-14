from __future__ import annotations
import numpy as np

from . import Attitude


class DirectionCosineMatrix(Attitude):
    def __init__(self, dcm: np.matrix) -> None:
        """
        Initialize the DirectionCosineMatrix class with a direction cosine matrix.

        :param dcm: A 3x3 direction cosine matrix.
        """
        super().__init__()
        assert dcm.shape == (3, 3), "DCM must be a 3x3 matrix"
        assert np.isclose(dcm @ dcm.T, np.eye(3)).all(), "DCM must be orthogonal"

        self.c11 = dcm[0, 0]
        self.c12 = dcm[0, 1]
        self.c13 = dcm[0, 2]
        self.c21 = dcm[1, 0]
        self.c22 = dcm[1, 1]
        self.c23 = dcm[1, 2]
        self.c31 = dcm[2, 0]
        self.c32 = dcm[2, 1]
        self.c33 = dcm[2, 2]
    
    def __matmul__(self, other: "DirectionCosineMatrix") -> "DirectionCosineMatrix":
        """
        Multiply two direction cosine matrices.

        :param other: Another instance of DirectionCosineMatrix.
        :return: A new instance of DirectionCosineMatrix representing the product.
        """
        assert isinstance(other, DirectionCosineMatrix), "Can only multiply with another DirectionCosineMatrix"
        
        result = self.dcm.get_matrix() @ other.dcm.get_matrix()
        return DirectionCosineMatrix(result)
    
    def get_matrix(self) -> np.matrix:
        """
        Get the direction cosine matrix.

        :return: The direction cosine matrix as an np matrix.
        """
        return np.mat([
            [self.c11, self.c12, self.c13],
            [self.c21, self.c22, self.c23],
            [self.c31, self.c32, self.c33]
        ])
    
    # conversion methods
    def to_quaternion(self) -> Quaternion:
        """
        Convert the direction cosine matrix to a quaternion.

        :return: An instance of Quaternion.
        """
        from . import Quaternion
        q4 = np.sqrt(1 + self.c11 + self.c22 + self.c33) / 2
        q1 = (self.c23 - self.c32) / (4 * q4)
        q2 = (self.c31 - self.c13) / (4 * q4)
        q3 = (self.c12 - self.c21) / (4 * q4)
        return Quaternion(q1, q2, q3, q4)

    def to_euler_angles(self) -> YawPitchRoll:
        """
        Convert the direction cosine matrix to Euler angles.
        """
        from . import YawPitchRoll
        yaw = np.arctan2(self.c21, self.c11)
        pitch = np.arcsin(-self.c31)
        roll = np.arctan2(self.c32, self.c33)
        return YawPitchRoll(yaw, pitch, roll)
    
    def to_eigenaxis(self) -> EigenaxisRotation:
        """
        Convert the direction cosine matrix to an eigenaxis rotation.

        :return: An instance of EigenaxisRotation.
        """
        from . import EigenaxisRotation
        eigenangle = np.arccos((self.c11 + self.c22 + self.c33 - 1) / 2)
        e1 = (self.c23 - self.c32) / (2 * np.sin(eigenangle))
        e2 = (self.c31 - self.c13) / (2 * np.sin(eigenangle))
        e3 = (self.c12 - self.c21) / (2 * np.sin(eigenangle))
        return EigenaxisRotation(eigenangle, np.array([e1, e2, e3]))

    @staticmethod
    def calculate_t_1(angle: float) -> DirectionCosineMatrix:
        """
        Calculate the direction cosine matrix for a rotation about the x-axis.
        """
        return DirectionCosineMatrix(np.asmatrix([
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]
        ]))
    
    @staticmethod
    def calculate_t_2(angle: float) -> DirectionCosineMatrix:
        """
        Calculate the direction cosine matrix for a rotation about the y-axis.
        """
        return DirectionCosineMatrix(np.asmatrix([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ]))
    
    @staticmethod
    def calculate_t_3(angle: float) -> DirectionCosineMatrix:
        """
        Calculate the direction cosine matrix for a rotation about the z-axis.
        """
        return DirectionCosineMatrix(np.asmatrix([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ]))
