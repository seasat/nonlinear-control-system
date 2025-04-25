import numpy as np


class Spacecraft:
    def __init__(self, inertia_tensor: np.matrix) -> None:
        """
        Initialize the Spacecraft class with an inertia tensor.

        :param inertia_tensor: The inertia tensor of the spacecraft.
        """
        assert inertia_tensor.shape == (3, 3), "Inertia tensor must be a 3x3 matrix"

        self.inertia_tensor = inertia_tensor
