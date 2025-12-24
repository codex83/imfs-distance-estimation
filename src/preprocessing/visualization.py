"""
Preprocessing Visualization Functions

Functions to visualize preprocessing stages for debugging and presentation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .image_preprocessing import (
    undistort_image, apply_white_balance, apply_clahe, apply_gamma_correction
)
from ..config import config


def visualize_preprocessing_stages(image_path: str, fx: float = None, fy: float = None,
                                   dist_coeffs: np.ndarray = None, save_path: Optional[str] = None):
    """
    Visualize each preprocessing stage individually.
    
    Args:
        image_path: Path to input image
        fx, fy: Focal lengths (if None, uses config)
        dist_coeffs: Distortion coefficients (if None, uses config)
        save_path: Optional path to save the visualization
    """
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = original.shape[:2]
    
    # Get parameters from config if not provided
    if fx is None:
        fx = config.CAMERA['fx']
    if fy is None:
        fy = config.CAMERA['fy']
    if dist_coeffs is None:
        dist_coeffs = config.CAMERA['dist_coeffs']
    
    # Build camera matrix
    camera_matrix = config.get_camera_matrix(w, h)
    
    # Stage 1: Undistortion
    undistorted, new_camera_matrix = undistort_image(
        original, camera_matrix, dist_coeffs,
        alpha=config.PREPROCESSING['undistort_alpha']
    )
    
    # Stage 2: Undistortion + White Balance
    white_balanced = apply_white_balance(undistorted)
    
    # Stage 3: Undistortion + WB + CLAHE
    clahe_applied = apply_clahe(
        white_balanced,
        clip_limit=config.PREPROCESSING['clahe_clip_limit'],
        tile_grid_size=config.PREPROCESSING['clahe_tile_grid_size']
    )
    
    # Stage 4: Full pipeline (+ Gamma)
    gamma_applied = apply_gamma_correction(
        clahe_applied,
        gamma=config.PREPROCESSING['gamma']
    )
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Preprocessing Pipeline Stages', fontsize=16, fontweight='bold')
    
    # Row 1: Original -> Undistortion
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Stage 0: Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Stage 1: Undistortion\n(alpha={config.PREPROCESSING["undistort_alpha"]})',
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(white_balanced, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Stage 2: + White Balance\n(Gray World Algorithm)',
                        fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: CLAHE -> Gamma -> Final
    axes[1, 0].imshow(cv2.cvtColor(clahe_applied, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Stage 3: + CLAHE\n(clip_limit={config.PREPROCESSING["clahe_clip_limit"]}, '
                        f'tile={config.PREPROCESSING["clahe_tile_grid_size"]})',
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(gamma_applied, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Stage 4: + Gamma Correction\n(gamma={config.PREPROCESSING["gamma"]})',
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Comparison: Before and After
    comparison = np.hstack([original, gamma_applied])
    axes[1, 2].imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Before vs After\n(Left: Original | Right: Final)',
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("="*70)
    print("PREPROCESSING STATISTICS")
    print("="*70)
    print(f"Original size: {original.shape}")
    print(f"Undistorted size: {undistorted.shape}")
    print(f"New camera matrix:\n{new_camera_matrix}")
    print(f"\nNew focal lengths: fx={new_camera_matrix[0,0]:.2f}, fy={new_camera_matrix[1,1]:.2f}")
    print(f"New principal point: cx={new_camera_matrix[0,2]:.2f}, cy={new_camera_matrix[1,2]:.2f}")
    print("="*70)


def visualize_undistortion_with_grid(image_path: str, fx: float = None, fy: float = None,
                                     dist_coeffs: np.ndarray = None, save_path: Optional[str] = None):
    """
    Visualize undistortion with grid overlay to show the warping effect.
    
    Args:
        image_path: Path to input image
        fx, fy: Focal lengths (if None, uses config)
        dist_coeffs: Distortion coefficients (if None, uses config)
        save_path: Optional path to save the visualization
    """
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = original.shape[:2]
    
    # Get parameters from config if not provided
    if fx is None:
        fx = config.CAMERA['fx']
    if fy is None:
        fy = config.CAMERA['fy']
    if dist_coeffs is None:
        dist_coeffs = config.CAMERA['dist_coeffs']
    
    # Build camera matrix
    camera_matrix = config.get_camera_matrix(w, h)
    
    # Undistort
    undistorted, new_camera_matrix = undistort_image(
        original, camera_matrix, dist_coeffs,
        alpha=config.PREPROCESSING['undistort_alpha']
    )
    
    # Create grid overlay on original
    original_with_grid = original.copy()
    grid_spacing = 100
    
    # Draw horizontal lines
    for i in range(0, h, grid_spacing):
        cv2.line(original_with_grid, (0, i), (w, i), (0, 255, 0), 2)
    
    # Draw vertical lines
    for j in range(0, w, grid_spacing):
        cv2.line(original_with_grid, (j, 0), (j, h), (0, 255, 0), 2)
    
    # Create grid overlay on undistorted
    undistorted_with_grid = undistorted.copy()
    h_und, w_und = undistorted.shape[:2]
    
    # Draw horizontal lines
    for i in range(0, h_und, grid_spacing):
        cv2.line(undistorted_with_grid, (0, i), (w_und, i), (0, 255, 0), 2)
    
    # Draw vertical lines
    for j in range(0, w_und, grid_spacing):
        cv2.line(undistorted_with_grid, (j, 0), (j, w_und), (0, 255, 0), 2)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Lens Distortion Correction with Grid Visualization',
                 fontsize=16, fontweight='bold')
    
    # Left: Original with grid (showing distortion curves)
    axes[0].imshow(cv2.cvtColor(original_with_grid, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image\n(Green grid shows distortion curves)',
                     fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Undistorted with grid (showing straight lines)
    axes[1].imshow(cv2.cvtColor(undistorted_with_grid, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After Undistortion\n(Grid lines are now straight)',
                     fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("="*70)
    print("UNDISTORTION VISUALIZATION")
    print("="*70)
    print(f"Original image size: {original.shape}")
    print(f"Undistorted image size: {undistorted.shape}")
    print(f"\nDistortion coefficients:")
    print(f"  k1 (radial): {dist_coeffs[0]:.6f}")
    print(f"  k2 (radial): {dist_coeffs[1]:.6f}")
    print(f"  p1 (tangential): {dist_coeffs[2]:.6f}")
    print(f"  p2 (tangential): {dist_coeffs[3]:.6f}")
    print(f"  k3 (radial): {dist_coeffs[4]:.6f}")
    print(f"\nCamera Matrix (Original):")
    print(f"  fx: {fx:.2f}, fy: {fy:.2f}")
    print(f"  cx: {w/2:.2f}, cy: {h/2:.2f}")
    print(f"\nCamera Matrix (After Undistortion):")
    print(f"  fx: {new_camera_matrix[0,0]:.2f}, fy: {new_camera_matrix[1,1]:.2f}")
    print(f"  cx: {new_camera_matrix[0,2]:.2f}, cy: {new_camera_matrix[1,2]:.2f}")
    print("="*70)


def visualize_undistortion_comparison(image_path: str, fx: float = None, fy: float = None,
                                      dist_coeffs: np.ndarray = None, save_path: Optional[str] = None):
    """
    Visualize undistortion - shows original distorted vs corrected image.
    
    Args:
        image_path: Path to input image
        fx, fy: Focal lengths (if None, uses config)
        dist_coeffs: Distortion coefficients (if None, uses config)
        save_path: Optional path to save the visualization
    """
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = original.shape[:2]
    
    # Get parameters from config if not provided
    if fx is None:
        fx = config.CAMERA['fx']
    if fy is None:
        fy = config.CAMERA['fy']
    if dist_coeffs is None:
        dist_coeffs = config.CAMERA['dist_coeffs']
    
    # Build camera matrix
    camera_matrix = config.get_camera_matrix(w, h)
    
    # Undistort
    undistorted, new_camera_matrix = undistort_image(
        original, camera_matrix, dist_coeffs,
        alpha=config.PREPROCESSING['undistort_alpha']
    )
    
    # Resize to same height for comparison
    h_und, w_und = undistorted.shape[:2]
    if h_und != h:
        undistorted_resized = cv2.resize(undistorted, (w, h))
    else:
        undistorted_resized = undistorted
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Lens Distortion Correction', fontsize=16, fontweight='bold')
    
    # Left: Original (distorted)
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image\n(With Lens Distortion)',
                     fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Undistorted
    axes[1].imshow(cv2.cvtColor(undistorted_resized, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After Undistortion\n(Corrected)',
                     fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("="*70)
    print("UNDISTORTION COMPARISON")
    print("="*70)
    print(f"Original image size: {original.shape}")
    print(f"Undistorted image size: {undistorted.shape}")
    print(f"\nNotice:")
    print(f"  • Left: Curved/distorted road lines and horizon")
    print(f"  • Right: Straight road lines and horizon after correction")
    print("="*70)

