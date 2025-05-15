import numpy as np

from attitude.quaternion import Attitude


class Spacecraft:
    def __init__(self, inertia_tensor: np.matrix, attitude: Attitude, angular_velocity: np.matrix) -> None:
        """
        Initialize the Spacecraft class with an inertia tensor and attitude in form of a quaternion.

        :param inertia_tensor: The inertia tensor of the spacecraft.
        """
        assert inertia_tensor.shape == (3, 3), "Inertia tensor must be a 3x3 matrix"
        assert isinstance(attitude, Attitude), "Attitude must be an Attitude object"
        assert attitude.shape == (3, 1), "Attitude must be a 3x1 matrix"

        self.inertia_tensor = inertia_tensor
        self.attitude = attitude
        self.angular_velocity = angular_velocity
