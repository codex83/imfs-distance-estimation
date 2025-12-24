"""
Distance Estimation Models

This package contains the model architectures for the Intelligent Monocular Forward-Vision System (IMFS):
- HeavyweightModel: Heavy model with EfficientNetB4 + EfficientNetB3 backbones (main version)
- HeavyweightModelV2: Heavy model variant with EfficientNetB4 + MobileNetV2 backbones
- LightweightModel: Lightweight model with MobileNetV3-Small backbones
"""

from .heavyweight_model import HeavyweightModel
from .heavyweight_model_v2 import HeavyweightModelV2
from .lightweight_model import LightweightModel

__all__ = ['HeavyweightModel', 'HeavyweightModelV2', 'LightweightModel']



