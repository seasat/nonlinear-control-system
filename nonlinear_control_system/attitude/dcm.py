import numpy as np

from attitude import Attitude
from eigenaxis import EigenaxisRotation
from quaternion import Quaternion
from mrp import ModifiedRodriguezParameter


class DirectionCosineMatrix(Attitude):
    def __init__(self, dcm: np.mat) -> None:
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
    
    def __init__(self, q: Quaternion) -> None:
        """
        Initialize the DirectionCosineMatrix class with a quaternion.

        :param quaternion: An instance of Quaternion.
        """
        super().__init__()
        
        self.c11 = 1 - 2 * (q.q2**2 + q.q3**2)
        self.c12 = 2 * (q.q1 * q.q2 + q.q3 * q.q4)
        self.c13 = 2 * (q.q1 * q.q3 - q.q2 * q.q4)
        self.c21 = 2 * (q.q1 * q.q2 - q.q3 * q.q4)
        self.c22 = 1 - 2 * (q.q1**2 + q.q3**2)
        self.c23 = 2 * (q.q2 * q.q3 + q.q1 * q.q4)
        self.c31 = 2 * (q.q1 * q.q3 + q.q2 * q.q4)
        self.c32 = 2 * (q.q2 * q.q3 - q.q1 * q.q4)
        self.c33 = 1 - 2 * (q.q1**2 + q.q2**2)
    
    def __init__(self, mrp: ModifiedRodriguezParameter) -> None:
        """
        Initialize the DirectionCosineMatrix class with modified Rodriguez parameters.

        :param modified_rodriguez_parameter: An instance of ModifiedRodriguezParameter.
        """
        super().__init__()
        
        self.c11 = ((1 + mrp.squared())**2 - 8 * mrp.sigma2**2 - 8 * mrp.sigma3**2) / (1 + mrp.squared())**2
        self.c12 = 8 * mrp.sigma1 * mrp.sigma2 + 4 * mrp.sigma3 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        self.c13 = 8 * mrp.sigma1 * mrp.sigma3 + 4 * mrp.sigma2 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        self.c21 = 8 * mrp.sigma2 * mrp.sigma1 + 4 * mrp.sigma3 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        self.c22 = ((1 + mrp.squared())**2 - 8 * mrp.sigma1**2 - 8 * mrp.sigma3**2) / (1 + mrp.squared())**2
        self.c23 = 8 * mrp.sigma2 * mrp.sigma3 + 4 * mrp.sigma1 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        self.c31 = 8 * mrp.sigma3 * mrp.sigma1 + 4 * mrp.sigma2 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        self.c32 = 8 * mrp.sigma3 * mrp.sigma2 + 4 * mrp.sigma1 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        self.c33 = ((1 + mrp.squared())**2 - 8 * mrp.sigma1**2 - 8 * mrp.sigma2**2) / (1 + mrp.squared())**2
