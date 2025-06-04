from __future__ import annotations
import numpy as np


class Attitude:
    def __init__(self) -> None:
        pass

    def calculate_derivative(self, body_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        """
        Calculate the derivative of the attitude based on body rates and mean motion.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def to_vector(self) -> np.ndarray:
        """
        Convert the attitude to a vector representation.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def __sub__(self, other: Attitude) -> Attitude:
        """
        Subtract two attitudes.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
