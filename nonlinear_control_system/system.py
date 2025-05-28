import numpy as np
import control

from spacecraft import Spacecraft


def get_linearized_system(spacecraft: Spacecraft) -> control.StateSpace:
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

def get_dynamically_inverted_system() -> control.StateSpace:
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
