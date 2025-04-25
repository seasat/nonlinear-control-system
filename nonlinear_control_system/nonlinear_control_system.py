# -*- coding: utf-8 -*-
import numpy as np

from spacecraft import Spacecraft
from quantity import Quantity as Qty

# settings
INERTIA_TENSOR = Qty.matrix(
    np.asmatrix([
        [124.531, 0.0, 0.0],
        [0.0, 124.586, 0.0],
        [0.0, 0.0, 1.704]
    ]), 'kg m2'
)


if __name__ == "__main__":
    sc = Spacecraft(INERTIA_TENSOR)
