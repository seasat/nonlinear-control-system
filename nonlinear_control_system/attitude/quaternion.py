import numpy as np

from attitude import Attitude
from dcm import DirectionCosineMatrix
from eigenaxis import EigenaxisRotation
from mrp import ModifiedRodriguezParameter


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
    
    def __init__(self, q1: "Quaternion", q2: "Quaternion") -> None:
        """
        Initialize the Quaternion class with two quaternions.

        :param q1: The first quaternion.
        :param q2: The second quaternion.
        """
        super().__init__()
        
        quaternion_matrix = np.mat([
            [q2.q4, q2.q3, -q2.q2, q2.q1],
            [-q2.q3, q2.q4, q2.q1, q2.q2],
            [q2.q2, -q2.q1, q2.q4, q2.q3],
            [-q2.q1, -q2.q2, -q2.q3, q2.q4]
        ])
        result = quaternion_matrix @ np.mat([q1.q1, q1.q2, q1.q3, q1.q4]).T

        self.q1 = result[0, 0]
        self.q2 = result[1, 0]
        self.q3 = result[2, 0]
        self.q4 = result[3, 0]

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
    
    def __init__(self, dcm: DirectionCosineMatrix) -> None:
        """
        Initialize the Quaternion class with a direction cosine matrix.

        :param dcm: An instance of DirectionCosineMatrix.
        """
        super().__init__()
        
        self.q4 = np.sqrt(1 + dcm.c11 + dcm.c22 + dcm.c33) / 2
        self.q1 = (dcm.c23 - dcm.c32) / (4 * self.q4)
        self.q2 = (dcm.c31 - dcm.c13) / (4 * self.q4)
        self.q3 = (dcm.c12 - dcm.c21) / (4 * self.q4)
    
    def __init__(self, mrp: ModifiedRodriguezParameter) -> None:
        """
        Initialize the Quaternion class with modified Rodriguez parameters.

        :param modified_rodriguez_parameter: An instance of ModifiedRodriguezParameter.
        """
        super().__init__()
        
        self.q4 = 1 - mrp.squared() / (1 + mrp.squared()) 
        self.q1 = 2 * mrp.sigma1 / (1 + mrp.squared())
        self.q2 = 2 * mrp.sigma2 / (1 + mrp.squared())
        self.q3 = 2 * mrp.sigma3 / (1 + mrp.squared())
