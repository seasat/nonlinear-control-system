from collections.abc import Callable
import numpy as np

def rk4(derivative: Callable[[np.ndarray, float, ...], np.ndarray], state: np.ndarray, time: float, time_step: float, *args):
    """Runge-Kutta 4th order integration method."""
    k1 = derivative(state, time, *args)
    k2 = derivative(state + 0.5 * time_step * k1, time + 0.5 * time_step, *args)
    k3 = derivative(state + 0.5 * time_step * k2, time + 0.5 * time_step, *args)
    k4 = derivative(state + time_step * k3, time + time_step, *args)
    
    return state + (time_step / 6) * (k1 + 2*k2 + 2*k3 + k4)
