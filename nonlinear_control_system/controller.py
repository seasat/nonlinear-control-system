import numpy as np
import control

from spacecraft import Spacecraft
from attitude.angular_velocity import YPRRates, BodyRates
import dynamics


class Controller:
    def calculate_control_output(self, attitude_error: np.ndarray) -> np.ndarray:
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
        self.proportional_gains, self.derivative_gains = PDController.design_pd_controller(linear_plant, closed_loop_poles)
        self.get_closed_loop_system = PDController.get_closed_loop_system(linear_plant, np.hstack((self.proportional_gains, self.derivative_gains)))

    def calculate_control_output(self, attitude_error: np.ndarray) -> np.ndarray:
        """ Calculate the control torque based on the attitude and angular velocity errors. """
        control_variable_derivative = self.sc.angular_velocity
        control_output = -self.proportional_gains @ attitude_error + -self.derivative_gains @ control_variable_derivative
        return control_output

    @staticmethod
    def design_pd_controller(linear_system: control.StateSpace, desired_poles: list[complex]) -> tuple[np.ndarray, np.ndarray]:
        """Design a PD controller for a linear system using the pole placement method. """
        feedback_gains = control.place(linear_system.A, linear_system.B, desired_poles)
        proportional_gains = feedback_gains[:, 0:3] # np.diag([10, 10, 0.5])
        derivative_gains = feedback_gains[:, 3:6] # np.diag([50, 50, 1.2])
        return proportional_gains, derivative_gains

    @staticmethod
    def calculate_poles(inertia_tensor: np.ndarray, natural_frequency: float, damping_ratio: float) -> np.ndarray:
        pole = complex(-damping_ratio * natural_frequency, natural_frequency * np.sqrt(1 - damping_ratio**2))
        conjugate_pole = complex(pole.real, -pole.imag)
        poles = [pole, conjugate_pole] * 3

        return np.asarray(poles, dtype=complex)

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
        Get the state space representation for the linearized system
        dx/dt = A @ x + B @ u
        y = C @ x + D @ u
        with state vector x and output vector y.
        """
        a = np.zeros((6, 6))
        a[0:3, 3:6] = np.eye(3)  # Identity matrix for angular velocity
        a[0, 2] = spacecraft.orbit.mean_motion
    
        b = np.zeros((6, 3))
        # control torque affect the angular velocity
        b[3:6, 0:3] = np.linalg.inv(spacecraft.inertia_tensor)
    
        # track attitude
        c = np.zeros((6, 6))
        c[0:3, 0:3] = np.eye(3)
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

    def calculate_control_output(self, attitude_error: np.ndarray) -> np.ndarray:
        ypr_rates_state_derivative = self.sc.angular_velocity.calculate_ypr_rate_state_derivative(self.sc.attitude, self.sc.orbit.mean_motion) # d/dx (N(θ)*ω)

        target_ypr_accelerations = self.linear_controller.calculate_control_output(attitude_error) # virtual control output nu(x)
        current_ypr_accelerations = self._calculate_ypr_accelerations(self.sc, ypr_rates_state_derivative) # l(x)
        ypr_acceleration_error = target_ypr_accelerations - current_ypr_accelerations # nu(x) - l(x)

        torque_to_ypr_acceleration_matrix = self._calculate_dynamic_transfer_matrix(self.sc) # M(x)
        ypr_acceleration_to_control_torque_matrix = np.linalg.inv(torque_to_ypr_acceleration_matrix) # M(x)^-1
        control_torque = ypr_acceleration_to_control_torque_matrix @ ypr_acceleration_error # M(x)^-1 * (nu(x) - l(x))

        return control_torque
    
    def _calculate_ypr_accelerations(self, sc: Spacecraft, ypr_rates_state_derivative: np.ndarray) -> np.ndarray:
        ypr_rates = sc.angular_velocity.to_ypr_rates(self.sc.attitude, self.sc.orbit.mean_motion)
        angular_accelerations = dynamics.calculate_angular_acceleration(sc.angular_velocity, sc.inertia_tensor, self.disturbance_torque, sc.orbit.mean_motion)
        ypr_accelerations = ypr_rates_state_derivative @ np.vstack((ypr_rates, angular_accelerations))
        return ypr_accelerations
    
    def _calculate_dynamic_transfer_matrix(self, sc: Spacecraft) -> np.ndarray:
        ypr_rate_state_derivative = sc.angular_velocity.calculate_ypr_rate_state_derivative(self.sc.attitude, self.sc.orbit.mean_motion)  
        dynamic_transfer_matrix = ypr_rate_state_derivative @ np.vstack((np.zeros((3, 3)), self.j_inv))
        return dynamic_transfer_matrix

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

    def calculate_control_output(self, attitude_error: np.ndarray) -> np.ndarray:
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

    def calculate_control_output(self, attitude_error: np.ndarray) -> np.ndarray:
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
