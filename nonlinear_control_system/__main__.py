# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import control

from spacecraft import Spacecraft
from simulation import Simulation
from attitude import YawPitchRoll, AngularVelocity
from orbit import Orbit
from controller import Controller, PDController
import dynamics


def main():
    # settings
    ORBIT = Orbit(
        gravitational_parameter = 3.98600442e14, # m^3 s^-2
        semi_major_axis = 6.378137e6 + 700e3 # m
    )
    INERTIA_TENSOR = np.asmatrix([
        [124.531, 0.0, 0.0],
        [0.0, 124.586, 0.0],
        [0.0, 0.0, 1.704]
    ]) # kg m^2
    DISTURBANCE_TORQUE = np.asmatrix([
        [1e-4],
        [1e-4],
        [1e-4],
    ]) # Nm
    INITIAL_ATTITUDE = YawPitchRoll([np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)]) # rad
    SIMULATION_DURATION = 1000 # s
    SAMPLE_TIME = 0.1 # s
    ATTITUDE_COMMANDS = {
        0: YawPitchRoll([0, 0, 0]),
        100: YawPitchRoll([np.deg2rad(60), np.deg2rad(60), np.deg2rad(60)]),
        500.1: YawPitchRoll([np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(-60)]),
        900.1: YawPitchRoll([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    }
    NATURAL_FREQUENCY = .8 # rad/s
    DAMPING_RATIO = 0.95

    sc = Spacecraft(INERTIA_TENSOR, INITIAL_ATTITUDE, np.zeros((3, 1)), ORBIT)

    pd_controller = PDController(sc, NATURAL_FREQUENCY, DAMPING_RATIO)
    simulation = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, ATTITUDE_COMMANDS, pd_controller)
    simulation.plot_attitudes()
    simulation.plot_attitude_errors()

    plt.show()


if __name__ == "__main__":
    SystemExit(main())
