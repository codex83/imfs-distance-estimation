"""
Unified Configuration System

Central configuration for all components of the distance estimation system.
"""

import numpy as np
from pathlib import Path


class Config:
    """
    Unified configuration class for distance estimation system.
    
    All configuration parameters are centralized here for easy management.
    """
    
    # ===== Preprocessing Configuration =====
    PREPROCESSING = {
        'undistort_alpha': 0.0,  # 0 = crop black borders, 1 = keep all pixels
        'clahe_clip_limit': 2.0,
        'clahe_tile_grid_size': (8, 8),
        'gamma': 1.5,
    }
    
    # ===== Camera Calibration =====
    # These are the actual calibrated values from your camera
    # Can be overridden by providing values during initialization
    CAMERA = {
        'fx': 2429.865965,
        'fy': 2424.492001,
        'cx': 1192.584876,  # Will be recalculated as w/2 if use_dynamic_center=True
        'cy': 1015.978074,  # Will be recalculated as h/2 if use_dynamic_center=True
        'dist_coeffs': np.array([-0.393931, 0.185580, 0.000120, 0.000002, 0.000000], dtype=np.float32),
        'avg_car_height': 1.5,  # Average car height in meters
        'use_dynamic_center': True,  # If True, cx/cy calculated as image center
    }
    
    # ===== Detection Configuration =====
    DETECTION = {
        'model_name': 'yolov8m.pt',
        'confidence_threshold': 0.25,
        'hood_exclude_ratio': 0.09,  # Bottom portion to exclude (9% for inference, 20% for training)
        'device': None,  # None = auto-detect
    }
    
    # ===== Model Configuration =====
    MODEL = {
        'precision': 'float32',  # 'float32' or 'float16' for faster inference
        'device': None,  # None = auto-detect
    }
    
    # ===== Inference Configuration =====
    INFERENCE = {
        'full_img_size': (224, 224),
        'patch_size': (64, 64),
        'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet
        'normalize_std': [0.229, 0.224, 0.225],   # ImageNet
    }
    
    # ===== Video Processing Configuration =====
    VIDEO = {
        'target_fps': 15,
        'codec': 'mp4v',
    }
    
    # ===== Training Configuration =====
    TRAINING = {
        'batch_size': 64,
        'num_workers': 8,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'unfreeze_epoch': 10,  # Epoch to unfreeze backbones
    }
    
    @classmethod
    def get_camera_matrix(cls, img_width: int, img_height: int) -> np.ndarray:
        """
        Build camera matrix from intrinsics.
        
        Args:
            img_width: Image width
            img_height: Image height
            
        Returns:
            3x3 camera matrix
        """
        fx = cls.CAMERA['fx']
        fy = cls.CAMERA['fy']
        
        if cls.CAMERA['use_dynamic_center']:
            cx = img_width / 2.0
            cy = img_height / 2.0
        else:
            cx = cls.CAMERA['cx']
            cy = cls.CAMERA['cy']
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @classmethod
    def update_camera_params(cls, fx: float = None, fy: float = None,
                             cx: float = None, cy: float = None,
                             dist_coeffs: np.ndarray = None):
        """
        Update camera calibration parameters.
        
        Args:
            fx, fy: Focal lengths
            cx, cy: Principal point
            dist_coeffs: Distortion coefficients
        """
        if fx is not None:
            cls.CAMERA['fx'] = fx
        if fy is not None:
            cls.CAMERA['fy'] = fy
        if cx is not None:
            cls.CAMERA['cx'] = cx
        if cy is not None:
            cls.CAMERA['cy'] = cy
        if dist_coeffs is not None:
            cls.CAMERA['dist_coeffs'] = dist_coeffs


# Create global config instance
config = Config()

