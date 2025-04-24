

class Attitude:
    def __init__(self, quaternion: tuple) -> None:
        """
        Initialize the Attitude class with a quaternion.

        :param quaternion: A tuple representing the quaternion (q0, q1, q2, q3).
        """
        assert len(quaternion) == 4, "Quaternion must be a tuple of length 4"
        (self.q1, self.q2, self.q3, self.q4) = quaternion
