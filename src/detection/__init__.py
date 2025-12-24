"""
Lead Vehicle Detection Module

This package provides YOLOv8-based lead vehicle detection functionality.
"""

from .lead_detector import HybridLeadDetector, process_dashcam_images

__all__ = ['HybridLeadDetector', 'process_dashcam_images']



