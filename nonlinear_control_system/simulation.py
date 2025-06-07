import numpy as np
import matplotlib.pyplot as plt

from spacecraft import Spacecraft
from attitude import Attitude, YawPitchRoll
import dynamics, integrator
from controller import Controller
from state_vector import StateVector


class Simulation:
    def __init__(self,
            spacecraft: Spacecraft,
            duration: float,
            sample_time: float,
            external_torque: np.matrix,
            target_attitude_commands: dict[float: Attitude],
            controller: Controller
        ) -> None:
        """
        Initialize the Simulation class with a spacecraft, duration, and sample time.

        :param spacecraft: The spacecraft to be simulated.
        :param duration: The total duration of the simulation in seconds.
        :param sample_time: The time interval between each simulation step in seconds.
        :param target_attitudes: A dictionary of target attitudes with time as the key and attitude as the value.
        """
        assert isinstance(spacecraft, Spacecraft), "Spacecraft must be an instance of Spacecraft"
        assert isinstance(target_attitude_commands, dict), "Target attitudes must be a dictionary"
        assert isinstance(external_torque, np.matrix), "External torque must be a numpy matrix"
        assert isinstance(controller, Controller), "Controller must be an instance of Controller"

        self.spacecraft = spacecraft
        self.duration = duration
        self.sample_time = sample_time
        self.external_torque = external_torque
        self.controller = controller

        self.sample_points = int(duration // sample_time + 1) # include start and end
        self.times = np.linspace(0, duration, self.sample_points, endpoint=True)

        # initialize history arrays
        self.attitudes = np.zeros(self.sample_points, dtype=type(spacecraft.attitude)) # use same attitude type as spacecraft stores
        self.angular_velocities = np.zeros_like(self.attitudes)
        self.target_attitudes = np.zeros_like(self.attitudes)
        self.attitude_errors = np.zeros_like(self.attitudes)
        self.control_torques = np.zeros((self.sample_points, 3, 1))

        self._calculate_target_attitudes(target_attitude_commands)
        self._run_simulation()
        self._calculate_errors()
    
    def _calculate_target_attitudes(self, target_attitude_commands: dict[float: Attitude]) -> None:
        """
        Calculate the target attitudes based on the target attitude commands.
        """
        target_attitude_commands = sorted(target_attitude_commands.items())
        command_iterator = iter(target_attitude_commands)
        current_command_attitude = None
        command = next(command_iterator)
        for idx, time in enumerate(self.times):
            if time >= command[0]:
                current_command_attitude = command[1]
                try:
                    command = next(command_iterator)
                except StopIteration:
                    pass # no more commands, don't update

            self.target_attitudes[idx] = current_command_attitude
    
    def _run_simulation(self) -> None:
        for idx, time in enumerate(self.times):
            target_attitude: YawPitchRoll = self.target_attitudes[idx]

            # calculate control torque
            control_torque = self.controller.calculate_control_output(target_attitude)
            torque = self.external_torque + control_torque

            # integrate rotational dynamics
            attitude, angular_velocity = integrator.rk4(
                self.spacecraft.attitude,
                self.spacecraft.angular_velocity,
                self.sample_time,
                self.spacecraft.inertia_tensor,
                torque,
                self.spacecraft.orbit.mean_motion,
            )

            # update spacecraft state
            self.spacecraft.angular_velocity = angular_velocity
            self.spacecraft.attitude = attitude

            # log step data
            self.attitudes[idx] = self.spacecraft.attitude
            self.angular_velocities[idx] = angular_velocity
            self.control_torques[idx] = control_torque

    def _calculate_errors(self) -> None:
        for idx in range(len(self.attitude_errors)):
            error = self.attitudes[idx] - self.target_attitudes[idx]
            self.attitude_errors[idx] = error
    
    def plot_attitudes(self) -> None:
        """
        Plot the attitudes over time.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), tight_layout=True)

        rolls = np.array([attitude.roll for attitude in self.attitudes])
        ax.plot(self.times, rolls, label="Roll")
        pitches = np.array([attitude.pitch for attitude in self.attitudes])
        ax.plot(self.times, pitches, label="Pitch")
        yaws = np.array([attitude.yaw for attitude in self.attitudes])
        ax.plot(self.times, yaws, label="Yaw")

        #command_yaws = np.array([attitude.yaw for attitude in self.target_attitudes])
        #ax.plot(self.times, command_yaws, label="Command", linestyle='--', color='k')
        ax.set_prop_cycle(None)
        command_components_1 = np.array([attitude.to_vector()[0] for attitude in self.target_attitudes])
        ax.plot(self.times, command_components_1, label=f"Command --", linestyle='--')
        command_components_2 = np.array([attitude.to_vector()[1] for attitude in self.target_attitudes])
        ax.plot(self.times, command_components_2, label=f"Command --", linestyle='--')
        command_components_3 = np.array([attitude.to_vector()[2] for attitude in self.target_attitudes])
        ax.plot(self.times, command_components_3, label=f"Command --", linestyle='--')

        ax.legend()

        ax.set_xlabel(r"Time $[\mathrm{s}]$")
        ax.set_ylabel(r"Attitude $[\mathrm{rad}]$")
    
    def plot_attitude_errors(self) -> None:
        """ Plot the attitude errors over time. """
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), tight_layout=True)

        rolls = np.array([error.roll for error in self.attitude_errors])
        ax.plot(self.times, np.abs(rolls), label="Roll Error")
        pitches = np.array([error.pitch for error in self.attitude_errors])
        ax.plot(self.times, np.abs(pitches), label="Pitch Error")
        yaws = np.array([error.yaw for error in self.attitude_errors])
        ax.plot(self.times, np.abs(yaws), label="Yaw Error")

        ax.legend()
        ax.set_yscale('log')

        ax.set_xlabel(r"Time $[\mathrm{s}]$")
        ax.set_ylabel(r"Attitude Error $[\mathrm{rad}]$")
    
    def plot_control_torques(self) -> None:
        """ Plot the control torques over time. """
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5), tight_layout=True)

        ax.plot(self.times, self.control_torques[:, 0, 0], label=r"${T}_{c,1}$")
        ax.plot(self.times, self.control_torques[:, 1, 0], label=r"${T}_{c,2}$")
        ax.plot(self.times, self.control_torques[:, 2, 0], label=r"${T}_{c,3}$")

        magnitudes = np.linalg.norm(self.control_torques, axis=1)
        ax.plot(self.times, magnitudes, label=r"$|\mathbf{T}_c|$", linestyle='--', color='k')

        ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel(r"Time $[\mathrm{s}]$")
        ax.set_ylabel(r"Control Torque $[\mathrm{Nm}]$")
