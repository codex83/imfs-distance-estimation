"""
Camera Calibration Configuration

This file contains camera calibration parameters needed for image preprocessing.
These values are now managed through the unified config system (src/config.py).

This file is kept for backward compatibility. New code should use src.config.Config.
"""

import numpy as np
from ..config import config

# Backward compatibility: Export values from unified config
CameraMat = config.get_camera_matrix(2448, 2048)  # Default size, will be recalculated dynamically
DistCoeff = config.CAMERA['dist_coeffs']
UNDISTORT_ALPHA = config.PREPROCESSING['undistort_alpha']
CLAHE_CLIP_LIMIT = config.PREPROCESSING['clahe_clip_limit']
CLAHE_TILE_GRID_SIZE = config.PREPROCESSING['clahe_tile_grid_size']
GAMMA = config.PREPROCESSING['gamma']



