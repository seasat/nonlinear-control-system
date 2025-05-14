# -*- coding: utf-8 -*-
import numpy as np

from spacecraft import Spacecraft
from simulation import Simulation
from attitude import YawPitchRoll


def main():
    # settings
    INERTIA_TENSOR = np.asmatrix([
        [124.531, 0.0, 0.0],
        [0.0, 124.586, 0.0],
        [0.0, 0.0, 1.704]
    ])
    INITIAL_ATTITUDE = YawPitchRoll(np.deg2rad(30), np.deg2rad(30), np.deg2rad(30))
    SIMULATION_DURATION = 1000 # s
    SAMPLE_TIME = 0.1 # s

    sc = Spacecraft(INERTIA_TENSOR, INITIAL_ATTITUDE)
    simulation = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME)


if __name__ == "__main__":
    SystemExit(main())
