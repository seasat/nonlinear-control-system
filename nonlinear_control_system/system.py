import numpy as np
import control


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
