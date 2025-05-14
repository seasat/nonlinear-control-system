import numpy as np
import matplotlib.pyplot as plt

from spacecraft import Spacecraft
from attitude import Attitude, YawPitchRoll


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
        self._run_simulation()
    
    def _calculate_target_attitudes(self, target_attitude_commands: dict[float: Attitude]) -> None:
        """
        Calculate the target attitudes based on the target attitude commands.
        """
        target_attitude_commands = sorted(target_attitude_commands.items())
        command_iterator = iter(target_attitude_commands)
        current_command_attitude = None
        command = target_attitude_commands[0]
        for idx, time in enumerate(self.times):
            if time >= command[0]:
                current_command_attitude = command[1]
                try:
                    command = next(command_iterator)
                except StopIteration:
                    break

            self.target_attitudes[idx] = current_command_attitude
    
    def _run_simulation(self) -> None:
        for idx, time in enumerate(self.times):
            target_attitude = self.target_attitudes[idx]

            attitude = YawPitchRoll(0, 0, 0)

            self.attitudes[idx] = attitude
    
    def plot_attitudes(self) -> None:
        """
        Plot the attitudes over time.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), tight_layout=True)

        yaws = np.array([attitude.yaw for attitude in self.attitudes])
        ax.plot(self.times, yaws, label="Yaw")
        pitches = np.array([attitude.pitch for attitude in self.attitudes])
        ax.plot(self.times, pitches, label="Pitch")
        rolls = np.array([attitude.roll for attitude in self.attitudes])
        ax.plot(self.times, rolls, label="Roll")

        ax.set_color_cycle(None)
        command_yaws = np.array([attitude.yaw for attitude in self.target_attitudes])
        ax.plot(self.times, command_yaws, label="Command Yaw", linestyle='--')
        command_pitches = np.array([attitude.pitch for attitude in self.target_attitudes])
        ax.plot(self.times, command_pitches, label="Command Pitch", linestyle='--')
        command_rolls = np.array([attitude.roll for attitude in self.target_attitudes])
        ax.plot(self.times, command_rolls, label="Command Roll", linestyle='--')

        ax.set_xlabel("Time $[\mathrm{s}]$")
        ax.set_ylabel("Attitude $[\mathrm{rad}]$")
