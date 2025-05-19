# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from spacecraft import Spacecraft
from simulation import Simulation
from attitude import YawPitchRoll, AngularVelocity


def main():
    # settings
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
    INITIAL_ATTITUDE = YawPitchRoll(np.deg2rad(30), np.deg2rad(30), np.deg2rad(30))
    SIMULATION_DURATION = 1000 # s
    SAMPLE_TIME = 0.1 # s
    ATTITUDE_COMMANDS = {
        0: YawPitchRoll(0, 0, 0),
        100: YawPitchRoll(np.deg2rad(60), np.deg2rad(60), np.deg2rad(60)),
        500.1: YawPitchRoll(np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(-60)),
        900.1: YawPitchRoll(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))
    }

    sc = Spacecraft(INERTIA_TENSOR, INITIAL_ATTITUDE, AngularVelocity([0, 0, 0]))
    simulation = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, ATTITUDE_COMMANDS)
    
    simulation.plot_attitudes()
    plt.show()


if __name__ == "__main__":
    SystemExit(main())
