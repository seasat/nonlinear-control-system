import numpy as np

from attitude import Attitude
from dcm import DirectionCosineMatrix


class EigenaxisRotation(Attitude):
    def __init__(self, eigenangle: float, eigenaxis: np.array[3]) -> None:
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

    def __init__(self, dcm: DirectionCosineMatrix) -> None:
        """
        Initialize the EigenaxisRotation class with a direction cosine matrix.

        :param dcm: An instance of DirectionCosineMatrix.
        """
        super().__init__()
        
        self.eigenangle = np.arccos((dcm.c11 + dcm.c22 + dcm.c33 - 1) / 2)
        self.e1 = (dcm.c23 - dcm.c32) / (2 * np.sin(self.eigenangle))
        self.e2 = (dcm.c31 - dcm.c13) / (2 * np.sin(self.eigenangle))
        self.e3 = (dcm.c12 - dcm.c21) / (2 * np.sin(self.eigenangle))