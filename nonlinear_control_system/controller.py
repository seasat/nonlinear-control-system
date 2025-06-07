import numpy as np
import control

from spacecraft import Spacecraft
import dynamics
from attitude import Attitude, YawPitchRoll, Quaternion


class Controller:
    def calculate_control_output(self, target_attitude: Attitude) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    
class StateFeedbackController(Controller):
    """ State Feedback Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, natural_frequency: float, damping_ratio: float) -> None:
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"

        self.sc = spacecraft # for derivative calculation
        state_space = self.get_state_space()
        desired_poles = self.calculate_poles(natural_frequency, damping_ratio)
        self.gains = control.place(state_space.A, state_space.B, desired_poles)
        self.target_body_rates = np.zeros((3, 1))  # linearization point

    def calculate_control_output(self, target_attitude: Attitude) -> np.ndarray:
        """ Control law u = -K * x_e, where K is the feedback gain matrix and x_e is the state error. """
        attitude_error = self.sc.attitude.calculate_error(target_attitude)  # Δθ = θ - θ_d
        attitude_control_variables = attitude_error.to_vector()[0:3]  # take only first three components for control (relevant for quaternion case)
        body_rate_error = self.sc.angular_velocity - self.target_body_rates  # Δω = ω - ω_d
        state_error = np.vstack([attitude_control_variables, body_rate_error])  # Δx = [Δθ; Δω]

        control_output = -self.gains @ state_error
        return control_output
    
    def get_state_space(self) -> control.StateSpace:
        if isinstance(self.sc.attitude, YawPitchRoll):
            return self.get_nadir_linearized_ypr_state_space(self.sc)
        elif isinstance(self.sc.attitude, Quaternion):
            return self.get_nadir_linearized_quaternion_state_space(self.sc)
        else:
            raise ValueError("Unsupported attitude representation. Use YawPitchRoll or Quaternion.")
    
    def calculate_poles(self, natural_frequency: float, damping_ratio: float) -> list[complex]:
        """
        Calculate the poles for system with a characteristic formula of form
        s^2 + 2ζω_0 s + ω_0^2 = 0
        where ζ is the damping ratio and ω_0 is the natural frequency.
        """
        pole = complex(-damping_ratio * natural_frequency, natural_frequency * np.sqrt(1 - damping_ratio**2))
        conjugate_pole = complex(pole.real, -pole.imag)
        return [pole, conjugate_pole] * 3
    
    @staticmethod
    def get_state_vector(attitude: Attitude, body_rates: np.ndarray) -> np.ndarray:
        control_variables_attitude = attitude.to_vector()[0:3] # take only first three components for control (relevant for quaternion case)
        return np.vstack([control_variables_attitude, body_rates])

    @staticmethod
    def get_nadir_linearized_ypr_state_space(spacecraft: Spacecraft) -> control.StateSpace:
        """
        Get the state space representation for the linearized system
        dx/dt = A @ x + B @ u
        y = C @ x + D @ u
        with state vector x = [roll, pitch, yaw, omega1, omega2, omega3] and output vector y.
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
    
    @staticmethod
    def get_nadir_linearized_quaternion_state_space(spacecraft: Spacecraft) -> control.StateSpace:
        """
        Get state space representation for the linearized system
        dx/dt = A @ x + B @ u
        y = C @ x + D @ u
        with state vector x = [q1, q2, q3, omega1, omega2, omega3] and output vector y = [q1, q2, q3].
        linearized around nadir pointing x = [0, 0, 0, 0, 0, 0]
        as only first three quaternion components are used as control variables.
        """
        a = np.zeros((6, 6))
        a[0, 3] = 0.5  # dq1/dt = 0.5 * ω1
        a[1, 4] = 0.5  # dq2/dt = 0.5 * ω2
        a[2, 5] = 0.5  # dq3/dt = 0.5 * ω3
        
        b = np.zeros((6, 3))
        b[3:6, 0:3] = np.linalg.inv(spacecraft.inertia_tensor)

        c = np.zeros((3, 6))
        c[0:3, 0:3] = np.eye(3)  # quaternion output
        d = np.zeros((3, 3))  # no feedthrough

        return control.StateSpace(a, b, c, d)


class PDController(Controller):
    """ Proportional-Derivative (PD) Controller for spacecraft attitude control assuming a double integrator plant. """
    def __init__(self, spacecraft: Spacecraft, natural_frequency: float, damping_ratio: float) -> None:
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"

        self.sc = spacecraft
        self._calculate_gains(natural_frequency, damping_ratio)

    def calculate_control_output(self, target_attitude: Attitude) -> np.ndarray:
        """ Control law u = -K_d * x_e - K_p * x_e_dot, where K_d is the derivative gain and K_p is the proportional gain. """
        attitude_difference: Attitude =  self.sc.attitude.calculate_error(target_attitude) # Δθ = θ - θ_d
        attitude_error: np.ndarray = attitude_difference.to_vector()  # convert to vector for calculations
        attitude_derivative: np.ndarray = self.sc.attitude.calculate_derivative(self.sc.angular_velocity, self.sc.orbit.mean_motion)

        control_output = -self.derivative_gain @ attitude_derivative[0:3] - self.proportional_gain @ attitude_error[0:3] # only first three components are used as control variables
        return control_output
    
    def _calculate_gains(self, natural_frequency: float, damping_ratio: float) -> None:
        """
        Calculate the gains for the PD controller using pole placement.
        Characteristic polynomial of closed-loop system with double integrator plant is:
        s^2 + K_d * s + K_p = 0
        where K_d is the derivative gain and K_p is the proportional gain.
        Reference form for pole placement is:
        s^2 + 2ζω_0 s + ω_0^2 = 0
        where ζ is the damping ratio and ω_0 is the natural frequency.
        """
        # assuming decoupled control for each axis -> diagonal gain matrices
        K_p = np.eye(3) * natural_frequency**2  # Proportional gain
        K_d = np.eye(3) * 2 * damping_ratio * natural_frequency  # Derivative gain

        self.proportional_gain = K_p
        self.derivative_gain = K_d

    
class NDIController(Controller):
    """ Nonlinear Dynamic Inversion (NDI) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, disturbance_torque: np.ndarray, natural_frequency, damping_ratio) -> None:
        """ Initialize the NDIController class with a spacecraft and linear controller parameters. """
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"
        assert disturbance_torque.shape == (3, 1), "disturbance_torque must be a 3x1 matrix"

        self.sc = spacecraft
        self.disturbance_torque = disturbance_torque

        self.j_inv = np.linalg.inv(spacecraft.inertia_tensor)
        self.linear_controller = PDController(spacecraft, natural_frequency, damping_ratio)

    def calculate_control_output(self, target_attitude: Attitude) -> np.ndarray:
        attitude_rate_state_derivative = self.sc.attitude.calculate_derivative_state_derivative(self.sc.angular_velocity, self.sc.orbit.mean_motion) # d/dx (N(θ)*ω + n*b(θ))

        target_attitude_accelerations = self.linear_controller.calculate_control_output(target_attitude) # virtual control output nu(x)
        current_attitude_accelerations = self._calculate_attitude_accelerations(self.sc, attitude_rate_state_derivative) # l(x)
        attitude_acceleration_error = target_attitude_accelerations - current_attitude_accelerations[0:3] # nu(x) - l(x), only first three components are used as control variables

        torque_to_attitude_acceleration_matrix = self._calculate_dynamic_transfer_matrix(attitude_rate_state_derivative) # M(x)
        ypr_acceleration_to_control_torque_matrix = np.linalg.inv(torque_to_attitude_acceleration_matrix[0:3, :]) # M(x)^-1, only use angular acceleration on first three components of attitude rate
        control_torque = ypr_acceleration_to_control_torque_matrix @ attitude_acceleration_error # M(x)^-1 * (nu(x) - l(x))

        return control_torque
    
    def _calculate_attitude_accelerations(self, sc: Spacecraft, attitude_rates_state_derivative: np.ndarray) -> np.ndarray:
        attitude_rates = sc.attitude.calculate_derivative(sc.angular_velocity, sc.orbit.mean_motion) 
        angular_accelerations = dynamics.calculate_angular_acceleration(sc.angular_velocity, sc.inertia_tensor, self.disturbance_torque, sc.orbit.mean_motion)
        ypr_accelerations = attitude_rates_state_derivative @ np.vstack((attitude_rates, angular_accelerations))
        return ypr_accelerations
    
    def _calculate_dynamic_transfer_matrix(self, attitude_rates_state_derivative: np.ndarray) -> np.ndarray:
        dynamic_transfer_matrix = attitude_rates_state_derivative @ np.vstack((np.zeros((self.sc.attitude.vector_length, 3)), self.j_inv))
        return dynamic_transfer_matrix


class TSSController(Controller):
    """ Time-Scale Separation (TSS) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, disturbance_torque: np.ndarray, natural_frequency: float, damping_ratio: float) -> None:
        assert isinstance(spacecraft, Spacecraft), "spacecraft must be an instance of Spacecraft"
        assert disturbance_torque.shape == (3, 1), "disturbance_torque must be a 3x1 matrix"

        self.sc = spacecraft
        self.disturbance_torque = disturbance_torque

        self.J_INV = np.linalg.inv(spacecraft.inertia_tensor)
        self._calculate_gains(natural_frequency, damping_ratio)

    def calculate_control_output(self, target_attitude: Attitude) -> np.ndarray:
        attitude_error = self.sc.attitude.calculate_error(target_attitude)  # Δθ = θ_d - θ
        attitude_error = attitude_error.to_vector()  # convert to vector for calculations

        # outer loop
        target_ypr_rates: np.ndarray = -self.outer_loop_gains @ attitude_error
        target_angular_velocity = self.sc.attitude.derivative_to_body_rates(target_ypr_rates, self.sc.orbit.mean_motion)

        # inner loop
        angular_velocity_error = self.sc.angular_velocity - target_angular_velocity  # Δω = ω - ω_d
        target_angular_acceleration = -self.inner_loop_gains @ angular_velocity_error

        control_torque: np.matrix = self.sc.inertia_tensor @ target_angular_acceleration + np.cross(self.sc.angular_velocity.flatten(), (self.sc.inertia_tensor @ self.sc.angular_velocity).flatten()).reshape(3, 1) - self.disturbance_torque
        return np.asarray(control_torque)
    
    def _calculate_gains(self, natural_frequency: float, damping_ratio: float) -> None:
        """
        Calculate the gains for the TSS controller using pole placement.
        Characteristic polynomial of closed-loop system with double integrator plant is:
        s^2 + K_1 * s + K_1 * K_2 = 0
        where K_d is the derivative gain and K_p is the proportional gain.
        Reference form for pole placement is:
        s^2 + 2ζω_0 s + ω_0^2 = 0
        where ζ is the damping ratio and ω_0 is the natural frequency.
        """
        # assuming decoupled control for each axis -> diagonal gain matrices
        K_1 = np.eye(3) * 2 * natural_frequency * damping_ratio  # inner loop gain
        K_2 = np.eye(3) * natural_frequency / 2 / damping_ratio  # outer loop gain
        self.inner_loop_gains = K_1
        self.outer_loop_gains = K_2


class INDIController(TSSController):
    """ Inverse Nonlinear Dynamic Inversion (INDI) Controller for spacecraft attitude control. """
    def __init__(self, spacecraft: Spacecraft, disturbance_torque: np.ndarray, natural_frequency: float, damping_ratio: float) -> None:
        super().__init__(spacecraft, disturbance_torque, natural_frequency, damping_ratio)

        self.last_control_torque = np.zeros((3, 1))  # last control torque for incremental dynamic inversion

    def calculate_control_output(self, target_attitude: Attitude) -> np.ndarray:
        attitude_error = self.sc.attitude.calculate_error(target_attitude)  # Δθ = θ_d - θ
        attitude_error = attitude_error.to_vector()  # convert to vector for calculations

        # outer loop
        target_ypr_rates: np.ndarray = -self.outer_loop_gains @ attitude_error
        target_angular_velocity = self.sc.attitude.derivative_to_body_rates(target_ypr_rates, self.sc.orbit.mean_motion)

        # inner loop
        angular_velocity_error = self.sc.angular_velocity - target_angular_velocity
        target_angular_acceleration = -self.inner_loop_gains @ angular_velocity_error

        # measured angular acceleration
        measured_angular_acceleration = dynamics.calculate_angular_acceleration(
            self.sc.angular_velocity, 
            self.sc.inertia_tensor, 
            self.disturbance_torque, 
            self.sc.orbit.mean_motion
        )

        control_torque = self.sc.inertia_tensor @ (target_angular_acceleration - measured_angular_acceleration) + self.last_control_torque
        return control_torque

        # linear control input
        target_angular_acceleration = self.linear_controller.calculate_control_output(target_attitude) # virtual control output nu(x)
        measured_angular_acceleration = dynamics.calculate_angular_acceleration(
            self.sc.angular_velocity, 
            self.sc.inertia_tensor, 
            self.disturbance_torque, 
            self.sc.orbit.mean_motion
        ) # assuming ideal measurement of angular acceleration
        angular_acceleration_error = target_angular_acceleration - measured_angular_acceleration

        # incremental dynamic inversion
        control_torque = self.last_control_torque + self.j @ (target_angular_acceleration - self.last_angular_acceleration)

        # log torque for next iteration
        self.last_control_torque = control_torque

        return control_torque
