import numpy as np

from spacecraft import Spacecraft
from attitude import Attitude


class Simulation:
    def __init__(self, spacecraft: Spacecraft, duration: float, sample_time: float, target_attitude_commands: dict[float: Attitude]) -> None:
        """
        Initialize the Simulation class with a spacecraft, duration, and sample time.

        :param spacecraft: The spacecraft to be simulated.
        :param duration: The total duration of the simulation in seconds.
        :param sample_time: The time interval between each simulation step in seconds.
        :param target_attitudes: A dictionary of target attitudes with time as the key and attitude as the value.
        """
        assert isinstance(spacecraft, Spacecraft), "Spacecraft must be an instance of Spacecraft"
        assert isinstance(target_attitude_commands, dict), "Target attitudes must be a dictionary"

        self.spacecraft = spacecraft
        self.duration = duration
        self.sample_time = sample_time

        self.sample_points = int(duration // sample_time + 1) # include start and end
        self.times = np.linspace(0, duration, self.sample_points, endpoint=True)

        # initialize history arrays
        self.attitudes = np.zeros(self.sample_points, dtype=type(spacecraft.attitude)) # use same attitude type as spacecraft stores
        self.angular_velocities = np.zeros_like(self.attitudes)
        self.target_attitudes = np.zeros_like(self.attitudes)

        self._calculate_target_attitudes(target_attitude_commands)
    
    def _calculate_target_attitudes(self, target_attitude_commands: dict[float: Attitude]) -> None:
        """
        Calculate the target attitudes based on the target attitude commands.
        """
        target_attitude_commands = sorted(target_attitude_commands.items())
        command_iterator = iter(target_attitude_commands)
        current_command_attitude = 0
        for idx, time in enumerate(self.times):
            if time >= command[0]:
                current_command_attitude = command[1]
                try:
                    command = next(command_iterator)
                except StopIteration:
                    break

            self.target_attitudes[idx] = current_command_attitude
