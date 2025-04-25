import numpy as np

from .attitude import Attitude
import attitude

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
    
    # alternative constructors/conversions
    @classmethod
    def from_eigenaxis(cls, e: EigenaxisRotation) -> "DirectionCosineMatrix":
        """
        Initialize the DirectionCosineMatrix class with an eigenaxis rotation.

        :param eigenaxis_rotation: An instance of EigenaxisRotation.
        """
        super().__init__()
        
        c11 = np.cos(e.eigenangle) + e.e1**2 * (1 - np.cos(e.eigenangle))
        c12 = e.e1 * e.e2 * (1 - np.cos(e.eigenangle)) + e.e3 * np.sin(e.eigenangle)
        c13 = e.e1 * e.e3 * (1 - np.cos(e.eigenangle)) - e.e2 * np.sin(e.eigenangle)
        c21 = e.e2 * e.e1 * (1 - np.cos(e.eigenangle)) - e.e3 * np.sin(e.eigenangle)
        c22 = np.cos(e.eigenangle) + e.e2**2 * (1 - np.cos(e.eigenangle))
        c23 = e.e2 * e.e3 * (1 - np.cos(e.eigenangle)) + e.e1 * np.sin(e.eigenangle)
        c31 = e.e3 * e.e1 * (1 - np.cos(e.eigenangle)) + e.e2 * np.sin(e.eigenangle)
        c32 = e.e3 * e.e2 * (1 - np.cos(e.eigenangle)) - e.e1 * np.sin(e.eigenangle)
        c33 = np.cos(e.eigenangle) + e.e3**2 * (1 - np.cos(e.eigenangle))

        return cls(np.mat([
            [c11, c12, c13],
            [c21, c22, c23],
            [c31, c32, c33]
        ]))
    
    @classmethod
    def from_quaternion(cls, q: Quaternion) -> "DirectionCosineMatrix":
        """
        Initialize the DirectionCosineMatrix class with a quaternion.

        :param quaternion: An instance of Quaternion.
        """
        super().__init__()
        
        c11 = 1 - 2 * (q.q2**2 + q.q3**2)
        c12 = 2 * (q.q1 * q.q2 + q.q3 * q.q4)
        c13 = 2 * (q.q1 * q.q3 - q.q2 * q.q4)
        c21 = 2 * (q.q1 * q.q2 - q.q3 * q.q4)
        c22 = 1 - 2 * (q.q1**2 + q.q3**2)
        c23 = 2 * (q.q2 * q.q3 + q.q1 * q.q4)
        c31 = 2 * (q.q1 * q.q3 + q.q2 * q.q4)
        c32 = 2 * (q.q2 * q.q3 - q.q1 * q.q4)
        c33 = 1 - 2 * (q.q1**2 + q.q2**2)

        return cls(np.mat([
            [c11, c12, c13],
            [c21, c22, c23],
            [c31, c32, c33]
        ]))
    
    @classmethod
    def from_mrp(cls, mrp: ModifiedRodriguezParameter) -> "DirectionCosineMatrix":
        """
        Initialize the DirectionCosineMatrix class with modified Rodriguez parameters.

        :param modified_rodriguez_parameter: An instance of ModifiedRodriguezParameter.
        """
        super().__init__()
        
        c11 = ((1 + mrp.squared())**2 - 8 * mrp.sigma2**2 - 8 * mrp.sigma3**2) / (1 + mrp.squared())**2
        c12 = 8 * mrp.sigma1 * mrp.sigma2 + 4 * mrp.sigma3 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        c13 = 8 * mrp.sigma1 * mrp.sigma3 + 4 * mrp.sigma2 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        c21 = 8 * mrp.sigma2 * mrp.sigma1 + 4 * mrp.sigma3 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        c22 = ((1 + mrp.squared())**2 - 8 * mrp.sigma1**2 - 8 * mrp.sigma3**2) / (1 + mrp.squared())**2
        c23 = 8 * mrp.sigma2 * mrp.sigma3 + 4 * mrp.sigma1 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        c31 = 8 * mrp.sigma3 * mrp.sigma1 + 4 * mrp.sigma2 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        c32 = 8 * mrp.sigma3 * mrp.sigma2 + 4 * mrp.sigma1 * (1 - mrp.squared()) / (1 + mrp.squared())**2
        c33 = ((1 + mrp.squared())**2 - 8 * mrp.sigma1**2 - 8 * mrp.sigma2**2) / (1 + mrp.squared())**2

        return cls(np.mat([
            [c11, c12, c13],
            [c21, c22, c23],
            [c31, c32, c33]
        ]))
