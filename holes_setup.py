# Copyright (c) <2022>, <Polatucha16>

import numpy as np

from classes import *

# List of balls to avoid 
holes = [Ball(np.array([[ 0.52, 0.63,-0.64]]), 0.7), 
         Ball(np.array([[-0.72,-0.47, 0.42]]), 0.5),
         Ball(np.array([[-0.30,-0.70,-0.27]]), 0.9)
        ]