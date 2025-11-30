from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Attitude(ABC):
    vector_length: int = 0  # Length of the vector representation of the attitude
    symbol: str = '' # Symbol representing the attitude type, e.g., 'q' for quaternion, 'θ' for yaw pitch roll angles.

    def __init__(self) -> None:
        pass

    @abstractmethod
    def calculate_derivative(self, body_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        """ Calculate the derivative of the attitude based on body rates and mean motion.  """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def derivative_to_body_rates(self, attitude_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        """ Convert the derivative of the attitude to body rates.  """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def calculate_derivative_state_derivative(self, body_rates: np.ndarray, mean_motion: float) -> np.ndarray:
        """ Calculate the change of attitude derivative with respect to attitude rates and angular accelerations.  """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """ Convert the attitude to a vector representation.  """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def calculate_error(self, other: Attitude) -> Attitude:
        """ Calculate the error between this attitude and another attitude.  """
        raise NotImplementedError("This method should be implemented by subclasses.")
