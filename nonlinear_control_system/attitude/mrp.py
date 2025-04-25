import numpy as np

import attitude


class ClassicalRodriguezParameter:
    def __init__(self, tau1: float, tau2: float, tau3: float) -> None:
        """
        Initialize the ClassicalRodriguezParameter class with classical Rodriguez parameters.

        :param tau1: The first classical Rodriguez parameter.
        :param tau2: The second classical Rodriguez parameter.
        :param tau3: The third classical Rodriguez parameter.
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3

    @classmethod
    def from_quaternion(cls, q: "Quaternion") -> "ClassicalRodriguezParameter":
        """
        Initialize the ClassicalRodriguezParameter class with a quaternion.

        :param quaternion: An instance of Quaternion.
        """
        super().__init__()
        
        tau1 = q.q1 / q.q4
        tau2 = q.q2 / q.q4
        tau3 = q.q3 / q.q4
        return cls(tau1, tau2, tau3)


class ModifiedRodriguezParameter:
    def __init__(self, sigma1: float, sigma2: float, sigma3: float) -> None:
        """
        Initialize the ModifiedRodriguezParameter class with modified Rodriguez parameters.

        :param sigma1: The first modified Rodriguez parameter.
        :param sigma2: The second modified Rodriguez parameter.
        :param sigma3: The third modified Rodriguez parameter.
        """
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
    
    def squared(self) -> float:
        """
        Calculate the squared norm of the modified Rodriguez parameter.

        :return: The squared norm of the modified Rodriguez parameter.
        """
        return self.sigma1**2 + self.sigma2**2 + self.sigma3**2

    # conversion methods
    def to_quaternion(self) -> "Quaternion":
        """
        Convert the modified Rodriguez parameter to a quaternion.

        :return: An instance of Quaternion.
        """
        squared = self.squared()
        q4 = 1 - squared / (1 + squared) 
        q1 = 2 * self.sigma1 / (1 + squared)
        q2 = 2 * self.sigma2 / (1 + squared)
        q3 = 2 * self.sigma3 / (1 + squared)
        return attitude.Quaternion(q1, q2, q3, q4) 

    def to_dcm(self) -> "DirectionCosineMatrix":
        """
        Convert the modified Rodriguez parameter to a direction cosine matrix.

        :return: An instance of DirectionCosineMatrix.
        """
        squared = self.squared()
        c11 = ((1 + squared)**2 - 8 * self.sigma2**2 - 8 * self.sigma3**2) / (1 + squared)**2
        c12 = 8 * self.sigma1 * self.sigma2 + 4 * self.sigma3 * (1 - squared) / (1 + squared)**2
        c13 = 8 * self.sigma1 * self.sigma3 + 4 * self.sigma2 * (1 - squared) / (1 + squared)**2
        c21 = 8 * self.sigma2 * self.sigma1 + 4 * self.sigma3 * (1 - squared) / (1 + squared)**2
        c22 = ((1 + squared)**2 - 8 * self.sigma1**2 - 8 * self.sigma3**2) / (1 + squared)**2
        c23 = 8 * self.sigma2 * self.sigma3 + 4 * self.sigma1 * (1 - squared) / (1 + squared)**2
        c31 = 8 * self.sigma3 * self.sigma1 + 4 * self.sigma2 * (1 - squared) / (1 + squared)**2
        c32 = 8 * self.sigma3 * self.sigma2 + 4 * self.sigma1 * (1 - squared) / (1 + squared)**2
        c33 = ((1 + squared)**2 - 8 * self.sigma1**2 - 8 * self.sigma2**2) / (1 + squared)**2

        return attitude.DirectionCosineMatrix(np.asmatrix([
            [c11, c12, c13],
            [c21, c22, c23],
            [c31, c32, c33]
        ]))

    @classmethod
    def from_quaternion(cls, q: "Quaternion") -> "ModifiedRodriguezParameter":
        """
        Initialize the ModifiedRodriguezParameter class with a quaternion.

        :param quaternion: An instance of Quaternion.
        """
        super().__init__()
        
        sigma1 = q.q1 / (1 + q.q4)
        sigma2 = q.q2 / (1 + q.q4)
        sigma3 = q.q3 / (1 + q.q4)
        return cls(sigma1, sigma2, sigma3)
