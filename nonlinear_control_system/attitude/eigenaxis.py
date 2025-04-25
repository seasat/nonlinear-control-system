import numpy as np

from .attitude import Attitude
import attitude

class EigenaxisRotation(Attitude):
    def __init__(self, eigenangle: float, eigenaxis: np.ndarray[3]) -> None:
        """
        Initialize the EigenaxisRotation class with an eigenangle and eigenaxis.

        :param eigenangle: The angle of rotation about the eigenaxis in radians.
        :param eigenaxis: A 3-element array representing the eigenaxis of rotation.
        """
        super().__init__()
        assert len(eigenaxis) == 3, "Eigenaxis must must have 3 elements"
        assert np.isclose(np.linalg.norm(eigenaxis), 1), "Eigenaxis must be a unit vector"

        self.eigenangle = eigenangle
        (self.e1, self.e2, self.e3) = eigenaxis

    # conversion methods
    def to_quaternion(self) -> "Quaternion":
        """
        Convert the eigenaxis rotation to a quaternion.

        :return: An instance of Quaternion.
        """
        q1 = self.e1 * np.sin(self.eigenangle / 2)
        q2 = self.e2 * np.sin(self.eigenangle / 2)
        q3 = self.e3 * np.sin(self.eigenangle / 2)
        q4 = np.cos(self.eigenangle / 2)
        return attitude.Quaternion(q1, q2, q3, q4)
    
    def to_dcm(self) -> "DirectionCosineMatrix":
        """
        Convert the eigenaxis rotation to a direction cosine matrix.

        :return: An instance of DirectionCosineMatrix.
        """
        c11 = np.cos(self.eigenangle) + self.e1**2 * (1 - np.cos(self.eigenangle))
        c12 = self.e1 * self.e2 * (1 - np.cos(self.eigenangle)) + self.e3 * np.sin(self.eigenangle)
        c13 = self.e1 * self.e3 * (1 - np.cos(self.eigenangle)) - self.e2 * np.sin(self.eigenangle)
        c21 = self.e2 * self.e1 * (1 - np.cos(self.eigenangle)) - self.e3 * np.sin(self.eigenangle)
        c22 = np.cos(self.eigenangle) + self.e2**2 * (1 - np.cos(self.eigenangle))
        c23 = self.e2 * self.e3 * (1 - np.cos(self.eigenangle)) + self.e1 * np.sin(self.eigenangle)
        c31 = self.e3 * self.e1 * (1 - np.cos(self.eigenangle)) + self.e2 * np.sin(self.eigenangle)
        c32 = self.e3 * self.e2 * (1 - np.cos(self.eigenangle)) - self.e1 * np.sin(self.eigenangle)
        c33 = np.cos(self.eigenangle) + self.e3**2 * (1 - np.cos(self.eigenangle))

        return attitude.DirectionCosineMatrix(np.asmatrix([
            [c11, c12, c13],
            [c21, c22, c23],
            [c31, c32, c33]
        ]))
    
    def to_crp(self) -> "ClassicalRodriguezParameter":
        """
        Convert the eigenaxis rotation to a Classical Rodriguez parameter.

        :return: An instance of ClassicalRodriguezParameter.
        """
        tau1 = self.e1 * np.tan(self.eigenangle / 2)
        tau2 = self.e2 * np.tan(self.eigenangle / 2)
        tau3 = self.e3 * np.tan(self.eigenangle / 2)
        return attitude.ClassicalRodriguezParameter(tau1, tau2, tau3)
    
    def to_mrp(self) -> "ModifiedRodriguezParameter":
        """
        Convert the eigenaxis rotation to a modified Rodriguez parameter.

        :return: An instance of ModifiedRodriguezParameter.
        """
        sigma1 = self.e1 * np.tan(self.eigenangle / 4)
        sigma2 = self.e2 * np.tan(self.eigenangle / 4)
        sigma3 = self.e3 * np.tan(self.eigenangle / 4)
        return attitude.ModifiedRodriguezParameter(sigma1, sigma2, sigma3)
