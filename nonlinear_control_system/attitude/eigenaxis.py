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

    @classmethod
    def from_dcm(cls, dcm: "DirectionCosineMatrix") -> "EigenaxisRotation":
        """
        Initialize the EigenaxisRotation class with a direction cosine matrix.

        :param dcm: An instance of DirectionCosineMatrix.
        """
        super().__init__()
        
        eigenangle = np.arccos((dcm.c11 + dcm.c22 + dcm.c33 - 1) / 2)
        e1 = (dcm.c23 - dcm.c32) / (2 * np.sin(eigenangle))
        e2 = (dcm.c31 - dcm.c13) / (2 * np.sin(eigenangle))
        e3 = (dcm.c12 - dcm.c21) / (2 * np.sin(eigenangle))

        return cls(eigenangle, np.array([e1, e2, e3]))
