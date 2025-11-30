

class Orbit:
    def __init__(self, gravitational_parameter: float, semi_major_axis: float) -> None:
        self.gravitational_parameter = gravitational_parameter
        self.semi_major_axis = semi_major_axis

        self.calculate_mean_motion()
    
    def calculate_mean_motion(self):
        self.mean_motion = (
            self.gravitational_parameter * self.semi_major_axis ** -3
        ) ** 0.5
    
    @property
    def n(self):
        return self.mean_motion
