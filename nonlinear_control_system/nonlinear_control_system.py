# -*- coding: utf-8 -*-
import numpy as np

from spacecraft import Spacecraft

# settings
INERTIA_TENSOR = np.mat([
    [124.531, 0.0, 0.0],
    [0.0, 124.586, 0.0],
    [0.0, 0.0, 1.704]
])


if __name__ == "__main__":
    sc = Spacecraft(INERTIA_TENSOR)
