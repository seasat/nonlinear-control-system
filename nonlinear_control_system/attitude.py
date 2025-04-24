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
