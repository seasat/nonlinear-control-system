import numpy as np

from attitude import Attitude
from orbit import Orbit


class Spacecraft:
    def __init__(self, inertia_tensor: np.matrix, attitude: Attitude, angular_velocity: np.ndarray, orbit: Orbit) -> None:
        """
        Initialize the Spacecraft class with an inertia tensor and attitude in form of a quaternion.

        :param inertia_tensor: The inertia tensor of the spacecraft.
        """
        assert inertia_tensor.shape == (3, 3), "Inertia tensor must be a 3x3 matrix"
        assert isinstance(attitude, Attitude), "Attitude must be an Attitude object"
        assert angular_velocity.shape == (3, 1), "Angular velocity must be a 3x1 ndarray"
        assert isinstance(orbit, Orbit) or None, "Orbit must be an Orbit object"

        self.inertia_tensor = inertia_tensor
        self.attitude = attitude
        self.angular_velocity = angular_velocity
        self.orbit = orbit
    
    def set_state(self, attitude: Attitude, angular_velocity: np.ndarray) -> None:
        """
        Set the state of the spacecraft.

        :param attitude: The new attitude of the spacecraft.
        :param angular_velocity: The new angular velocity of the spacecraft.
        """
        assert isinstance(attitude, Attitude), "Attitude must be an Attitude object"
        assert angular_velocity.shape == (3, 1), "Angular velocity must be a 3x1 ndarray"

        self.attitude = attitude
        self.angular_velocity = angular_velocity
