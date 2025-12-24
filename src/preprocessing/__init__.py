"""
Image Preprocessing Module

This package provides functions to clean and enhance dashcam images for distance estimation.
"""

from .image_preprocessing import (
    preprocess_image,
    undistort_image,
    apply_white_balance,
    apply_clahe,
    apply_gamma_correction
)

__all__ = [
    'preprocess_image',
    'undistort_image',
    'apply_white_balance',
    'apply_clahe',
    'apply_gamma_correction'
]



