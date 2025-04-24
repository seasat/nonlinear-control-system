import numpy as np

class Attitude:
    def __init__(self) -> None:
        pass


class DirectionCosineMatrix(Attitude):
    def __init__(self, dcm: np.mat) -> None:
        """
        Initialize the DirectionCosineMatrix class with a direction cosine matrix.

        :param dcm: A 3x3 direction cosine matrix.
        """
        super().__init__()
        assert dcm.shape == (3, 3), "DCM must be a 3x3 matrix"

        self.c11 = dcm[0, 0]
        self.c12 = dcm[0, 1]
        self.c13 = dcm[0, 2]
        self.c21 = dcm[1, 0]
        self.c22 = dcm[1, 1]
        self.c23 = dcm[1, 2]
        self.c31 = dcm[2, 0]
        self.c32 = dcm[2, 1]
        self.c33 = dcm[2, 2]
    
    def __init__(self, e: EigenaxisRotation) -> None:
        """
        Initialize the DirectionCosineMatrix class with an eigenaxis rotation.

        :param eigenaxis_rotation: An instance of EigenaxisRotation.
        """
        super().__init__()
        
        self.c11 = np.cos(e.eigenangle) + e.e1**2 * (1 - np.cos(e.eigenangle))
        self.c12 = e.e1 * e.e2 * (1 - np.cos(e.eigenangle)) + e.e3 * np.sin(e.eigenangle)
        self.c13 = e.e1 * e.e3 * (1 - np.cos(e.eigenangle)) - e.e2 * np.sin(e.eigenangle)
        self.c21 = e.e2 * e.e1 * (1 - np.cos(e.eigenangle)) - e.e3 * np.sin(e.eigenangle)
        self.c22 = np.cos(e.eigenangle) + e.e2**2 * (1 - np.cos(e.eigenangle))
        self.c23 = e.e2 * e.e3 * (1 - np.cos(e.eigenangle)) + e.e1 * np.sin(e.eigenangle)
        self.c31 = e.e3 * e.e1 * (1 - np.cos(e.eigenangle)) + e.e2 * np.sin(e.eigenangle)
        self.c32 = e.e3 * e.e2 * (1 - np.cos(e.eigenangle)) - e.e1 * np.sin(e.eigenangle)
        self.c33 = np.cos(e.eigenangle) + e.e3**2 * (1 - np.cos(e.eigenangle))


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
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4

    def __init__(self, e: EigenaxisRotation) -> None:
        """
        Initialize the Quaternion class with an eigenaxis rotation.

        :param eigenaxis_rotation: An instance of EigenaxisRotation.
        """
        super().__init__()
        
        self.q1 = e.e1 * np.sin(e.eigenangle / 2)
        self.q2 = e.e2 * np.sin(e.eigenangle / 2)
        self.q3 = e.e3 * np.sin(e.eigenangle / 2)
        self.q4 = np.cos(e.eigenangle / 2)
