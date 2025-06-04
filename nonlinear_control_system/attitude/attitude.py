from . import BodyRates

class Attitude:
    def __init__(self) -> None:
        pass

    def calculate_derivative(self, body_rates: BodyRates, n: float) -> np.ndarray:
        """
        Calculate the derivative of the attitude based on body rates and mean motion.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def to_vector(self) -> np.ndarray:
        """
        Convert the attitude to a vector representation.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
