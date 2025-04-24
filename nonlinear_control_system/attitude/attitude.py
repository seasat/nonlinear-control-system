import numpy as np

class Attitude:
    def __init__(self) -> None:
        pass


class ClassicalRodriguezParameter(Attitude):
    def __init__(self, tau1: float, tau2: float, tau3: float) -> None:
        """
        Initialize the ClassicalRodriguezParameter class with classical Rodriguez parameters.

        :param tau1: The first classical Rodriguez parameter.
        :param tau2: The second classical Rodriguez parameter.
        :param tau3: The third classical Rodriguez parameter.
        """
        super().__init__()
        
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
    
    def __init__(self, e: EigenaxisRotation) -> None:
        """
        Initialize the ClassicalRodriguezParameter class with an eigenaxis rotation.

        :param eigenaxis_rotation: An instance of EigenaxisRotation.
        """
        super().__init__()
        
        self.tau1 = e.e1 * np.tan(e.eigenangle / 2)
        self.tau2 = e.e2 * np.tan(e.eigenangle / 2)
        self.tau3 = e.e3 * np.tan(e.eigenangle / 2)

    def __init__(self, q: Quaternion) -> None:
        """
        Initialize the ClassicalRodriguezParameter class with a quaternion.

        :param quaternion: An instance of Quaternion.
        """
        super().__init__()
        
        self.tau1 = q.q1 / q.q4
        self.tau2 = q.q2 / q.q4
        self.tau3 = q.q3 / q.q4


class ModifiedRodriguezParameter(Attitude):
    def __init__(self, sigma1: float, sigma2: float, sigma3: float) -> None:
        """
        Initialize the ModifiedRodriguezParameter class with modified Rodriguez parameters.

        :param sigma1: The first modified Rodriguez parameter.
        :param sigma2: The second modified Rodriguez parameter.
        :param sigma3: The third modified Rodriguez parameter.
        """
        super().__init__()
        
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
    
    def squared(self) -> float:
        """
        Calculate the squared norm of the modified Rodriguez parameter.

        :return: The squared norm of the modified Rodriguez parameter.
        """
        return self.sigma1**2 + self.sigma2**2 + self.sigma3**2
    
    def __init__(self, e: EigenaxisRotation) -> None:
        """
        Initialize the ModifiedRodriguezParameter class with an eigenaxis rotation.

        :param eigenaxis_rotation: An instance of EigenaxisRotation.
        """
        super().__init__()
        
        self.sigma1 = e.e1 * np.tan(e.eigenangle / 4)
        self.sigma2 = e.e2 * np.tan(e.eigenangle / 4)
        self.sigma3 = e.e3 * np.tan(e.eigenangle / 4)

    def __init__(self, q: Quaternion) -> None:
        """
        Initialize the ModifiedRodriguezParameter class with a quaternion.

        :param quaternion: An instance of Quaternion.
        """
        super().__init__()
        
        self.sigma1 = q.q1 / (1 + q.q4)
        self.sigma2 = q.q2 / (1 + q.q4)
        self.sigma3 = q.q3 / (1 + q.q4)
