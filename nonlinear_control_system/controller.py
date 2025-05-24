import numpy as np
import control

from spacecraft import Spacecraft
import dynamics


class Controller:
    def __init__(self, spacecraft: Spacecraft, natural_frequency: float, damping_ratio: float) -> None:
        """
        Initialize the Controller class with a spacecraft and controller parameters.
        
        :param spacecraft: The spacecraft to be controlled.
        :param natural_frequency: The natural frequency of the controller.
        :param damping_ratio: The damping ratio of the controller.
        """
        self.spacecraft = spacecraft
        self.natural_frequency = natural_frequency
        self.damping_ratio = damping_ratio

        self.linear_system = Controller.get_linearized_system(spacecraft.inertia_tensor, spacecraft.orbit.mean_motion)
        self.gains = Controller.design_pd_controller(self.linear_system, natural_frequency, damping_ratio)
        self.get_closed_loop_system = Controller.get_closed_loop_system(self.linear_system, self.gains)

    @staticmethod
    def get_linearized_system(inertia_tensor: np.matrix, mean_motion: float) -> control.StateSpace:
        """
        Get the system matrices A and B for the linearized system
        dy/dt = A @ y + B @ u
        """
        J_INV = np.linalg.inv(inertia_tensor)

        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)  # Identity matrix for angular velocity
        a[0, 1] = mean_motion
        orbital_coupling = np.array([
            [0, 0, -mean_motion * (inertia_tensor[2,2] + inertia_tensor[1,1])],
            [0, 0, 0],
            [mean_motion * (inertia_tensor[0,0] + inertia_tensor[1,1]), 0, 0],
        ])
        a[3:6, 3:6] = -np.linalg.inv(inertia_tensor) @ orbital_coupling

        b = np.zeros((6, 3))
        # control torque affect the angular velocity
        b[3:6, 0:3] = J_INV

        # track all outputs
        c = np.eye(6)
        # no feedthrough
        d = np.zeros((6, 3))

        return control.StateSpace(a, b, c, d)

    @staticmethod
    def design_pd_controller(linear_system: control.StateSpace, natural_frequency: float, damping_ratio: float) -> control.StateSpace:
        """Design a PD controller for a linear system."""
        pole = complex(-damping_ratio * natural_frequency, natural_frequency * np.sqrt(1 - damping_ratio**2))
        conjugate_pole = complex(pole.real, -pole.imag)
        desired_poles = [pole, conjugate_pole] * 3

        feedback_gains = control.place(linear_system.A, linear_system.B, desired_poles)
        return feedback_gains

    @staticmethod
    def get_closed_loop_system(open_loop_system: control.StateSpace, feedback_gains: np.matrix) -> control.StateSpace:
        """
        Get the closed-loop system by combining an open-loop system with a controller.
        dy/dt = (A - B*K) @ y + B @ u
        """
        closed_loop_a = open_loop_system.A - open_loop_system.B @ feedback_gains
        return control.StateSpace(closed_loop_a, open_loop_system.B, open_loop_system.C, open_loop_system.D)
