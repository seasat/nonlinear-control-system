# -*- coding: utf-8 -*-
import numpy as np

from spacecraft import Spacecraft
from quantity import Quantity as Qty
from simulation import Simulation


def main():
    # settings
    INERTIA_TENSOR = Qty.matrix(
        np.asmatrix([
            [124.531, 0.0, 0.0],
            [0.0, 124.586, 0.0],
            [0.0, 0.0, 1.704]
        ]), 'kg m2'
    )
    SIMULATION_DURATION = Qty(1000, 's')
    SAMPLE_TIME = Qty(0.1, 's')

    sc = Spacecraft(INERTIA_TENSOR, None)
    simulation = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME)


if __name__ == "__main__":
    SystemExit(main())
