import numpy as np
import control

from spacecraft import Spacecraft
from attitude.angular_velocity import YPRRates, BodyRates


class Controller:
    def calculate_control_torque(self, attitude_error: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    
class PDController(Controller):
    """ Proportional-Derivative (PD) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, linear_plant: control.StateSpace, closed_loop_poles: list[complex]) -> None:
        """
        Initialize the Controller class with a spacecraft and controller parameters.
        
        :param linear_plant: The linear state-space representation of the plant.
        :param closed_loop_poles: The desired closed-loop poles for the PD controlled system.
        """
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"

        self.sc = spacecraft # for derivative calculation
        self.gains = PDController.design_pd_controller(linear_plant, closed_loop_poles)
        self.get_closed_loop_system = PDController.get_closed_loop_system(linear_plant, self.gains)

    def calculate_control_output(self, attitude_error: np.ndarray) -> np.ndarray:
        """ Calculate the control torque based on the attitude and angular velocity errors. """
        control_variable_derivative = self.sc.angular_velocity.to_ypr_rates(self.sc.attitude, self.sc.orbit.mean_motion)
        control_output = self.gains[:, 0:3] @ attitude_error + self.gains[:, 3:6] @ control_variable_derivative
        return control_output
    
    @property
    def proportional_gain(self) -> np.ndarray:
        """ Get the proportional gain matrix from the controller gains. """
        return self.gains[:, 0:3]

    @property
    def derivative_gain(self) -> np.ndarray:
        """ Get the derivative gain matrix from the controller gains. """
        return self.gains[:, 3:6]

    @staticmethod
    def design_pd_controller(linear_system: control.StateSpace, desired_poles: list[complex]) -> control.StateSpace:
        """Design a PD controller for a linear system using the pole placement method. """
        feedback_gains = control.place(linear_system.A, linear_system.B, desired_poles)
        return feedback_gains

    @staticmethod
    def calculate_poles(inertia_tensor: np.ndarray, natural_frequency: float, damping_ratio: float) -> list:
        inertia_ratios = np.asarray([
            1,
            inertia_tensor[1, 1] / inertia_tensor[0, 0],
            inertia_tensor[2, 2] / inertia_tensor[0, 0]
        ])
        pole = complex(-damping_ratio * natural_frequency, natural_frequency * np.sqrt(1 - damping_ratio**2))
        conjugate_pole = complex(pole.real, -pole.imag)
        poles = [[pole * inertia_ratio, conjugate_pole * inertia_ratio] for inertia_ratio in inertia_ratios]
        poles = np.array(poles).flatten()

        return poles

    @staticmethod
    def get_closed_loop_system(open_loop_system: control.StateSpace, feedback_gains: np.matrix) -> control.StateSpace:
        """
        Get the closed-loop system by combining an open-loop system with a controller.
        dy/dt = (A - B*K) @ y + B @ u
        """
        closed_loop_a = open_loop_system.A - open_loop_system.B @ feedback_gains
        return control.StateSpace(closed_loop_a, open_loop_system.B, open_loop_system.C, open_loop_system.D)
    
    @staticmethod
    def get_system_model(spacecraft: Spacecraft) -> control.StateSpace:
        """
        Get the system matrices A and B for the linearized system
        dy/dt = A @ y + B @ u
        """
        J = spacecraft.inertia_tensor
        n = spacecraft.orbit.mean_motion
        J_INV = np.linalg.inv(J)
    
        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)  # Identity matrix for angular velocity
        a[0, 1] = n
        orbital_coupling = np.array([
            [0, 0, -n * (J[2,2] + J[1,1])],
            [0, 0, 0],
            [n * (J[0,0] + J[1,1]), 0, 0],
        ])
        a[3:6, 3:6] = -J_INV @ orbital_coupling
    
        b = np.zeros((6, 3))
        # control torque affect the angular velocity
        b[3:6, 0:3] = J_INV
    
        # track all outputs
        c = np.eye(6)
        # no feedthrough
        d = np.zeros((6, 3))
    
        return control.StateSpace(a, b, c, d)

    
class NDIController(Controller):
    """ Nonlinear Dynamic Inversion (NDI) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, disturbance_torque: np.ndarray, closed_loop_poles: list[complex]) -> None:
        """ Initialize the NDIController class with a spacecraft and linear controller parameters. """
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"
        assert disturbance_torque.shape == (3, 1), "disturbance_torque must be a 3x1 matrix"

        self.sc = spacecraft
        self.disturbance_torque = disturbance_torque

        self.j_inv = np.linalg.inv(spacecraft.inertia_tensor)
        self.linear_controller = PDController(spacecraft, self.get_system_model(), closed_loop_poles)

    def calculate_control_torque(self, attitude_error: np.ndarray) -> np.ndarray:
        virtual_control_output = self.linear_controller.calculate_control_torque(attitude_error)
        ypr_rates_derivative = self.sc.angular_velocity.calculate_ypr_rate_derivative(self.sc.attitude, self.sc.orbit.mean_motion)

        transform_matrix = ypr_rates_derivative @ np.vstack((np.zeros((3, 3)), self.j_inv))
        inversion_offset = self._calculate_inversion_offset(ypr_rates_derivative, transform_matrix)
        return np.linalg.inv(transform_matrix) @ (virtual_control_output - inversion_offset)
    
    def _calculate_inversion_offset(self, ypr_rates_derivative: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        angular_velocity = self.sc.angular_velocity

        ypr_rates = self.sc.angular_velocity.to_ypr_rates(self.sc.attitude, self.sc.orbit.mean_motion)
        accelerations = -self.j_inv @ np.cross(angular_velocity.flatten(), (self.sc.inertia_tensor @ angular_velocity).flatten()).reshape(3, 1)

        return ypr_rates_derivative @ np.vstack((ypr_rates, accelerations)) + transform_matrix @ self.disturbance_torque
    
    @staticmethod
    def get_system_model() -> control.StateSpace:
        """
        Get state space of double integrator system after dynamic inversion where plant is
        G(s) = 1/s^2
        """
        # double integrator system
        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)  # Identity matrix for angular velocity

        b = np.zeros((6, 3))
        b[3:6, 0:3] = np.eye(3)  # Control torque affects the angular velocity

        c = np.eye(6)  # Track all outputs
        d = np.zeros((6, 3))  # No feedthrough
        
        return control.StateSpace(a, b, c, d)


class TSSController(Controller):
    """ Time-Scale Separation (TSS) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, disturbance_torque: np.ndarray, closed_loop_poles: list[complex]) -> None:
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"
        assert disturbance_torque.shape == (3, 1), "disturbance_torque must be a 3x1 matrix"

        self.sc = spacecraft
        self.disturbance_torque = disturbance_torque

        self.J_INV = np.linalg.inv(spacecraft.inertia_tensor)
        self.linear_controller = PDController(spacecraft, self.get_system_model(), closed_loop_poles)

    def calculate_control_torque(self, attitude_error: np.ndarray) -> np.ndarray:
        # outer loop
        target_ypr_rates = YPRRates(self.linear_controller.proportional_gain @ attitude_error)
        target_angular_velocity = target_ypr_rates.to_body_rates(self.sc.attitude, self.sc.orbit.mean_motion)

        # inner loop
        angular_velocity_error = target_angular_velocity - self.sc.angular_velocity
        target_angular_acceleration = self.linear_controller.derivative_gain @ angular_velocity_error

        control_torque = self.sc.inertia_tensor @ target_angular_acceleration + np.cross(self.sc.angular_velocity.flatten(), (self.sc.inertia_tensor @ self.sc.angular_velocity).flatten()).reshape(3, 1) - self.disturbance_torque
        return control_torque
    
    @staticmethod
    def get_system_model() -> control.StateSpace:
        """
        Get state space of double integrator system after dynamic inversion where plant is
        G(s) = 1/s^2
        """
        # double integrator system
        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)  # Identity matrix for angular velocity

        b = np.zeros((6, 3))
        b[3:6, 0:3] = np.eye(3)  # Control torque affects the angular velocity

        c = np.eye(6)  # Track all outputs
        d = np.zeros((6, 3))  # No feedthrough
        
        return control.StateSpace(a, b, c, d)


class INDIController(Controller):
    """ Inverse Nonlinear Dynamic Inversion (INDI) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, disturbance_torque: np.ndarray, closed_loop_poles: list[complex]) -> None:
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"
        assert disturbance_torque.shape == (3, 1), "disturbance_torque must be a 3x1 matrix"

        self.sc = spacecraft
        self.disturbance_torque = disturbance_torque
        self.last_control_torque = np.zeros((3, 1))
        self.last_angular_acceleration = np.zeros((3, 1))

        self.linear_controller = PDController(spacecraft, self.get_system_model(), closed_loop_poles)

    def calculate_control_torque(self, attitude_error: np.ndarray) -> np.ndarray:
        # outer loop
        target_ypr_rates: YPRRates = YPRRates(self.linear_controller.proportional_gain @ attitude_error)
        target_body_rates: BodyRates = target_ypr_rates.to_body_rates(self.sc.attitude, self.sc.orbit.mean_motion)

        # inner loop
        body_rate_error = target_body_rates - self.sc.angular_velocity # control variable
        target_angular_acceleration = self.linear_controller.derivative_gain @ body_rate_error # virtual control output
        control_torque_increment = self.sc.inertia_tensor @ (target_angular_acceleration - self.last_angular_acceleration) # dynamic inversion
        control_torque = self.last_control_torque + control_torque_increment # incremental control

        # log reference values for linearization at next step
        self.last_angular_acceleration = target_angular_acceleration
        self.last_control_torque = control_torque
        return control_torque

    @staticmethod
    def get_system_model() -> control.StateSpace:
        """
        Get state space of double integrator system after dynamic inversion where plant is
        G(s) = 1/s^2
        """
        # double integrator system
        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)  # Identity matrix

        b = np.zeros((6, 3))
        b[3:6, 0:3] = np.eye(3)

        c = np.eye(6)  # Track all outputs
        d = np.zeros((6, 3))

        return control.StateSpace(a, b, c, d)
