"""
Image Preprocessing Pipeline

This module provides functions to clean and enhance dashcam images.
It is designed to be both:
  1. A standalone script to batch-process folders.
  2. A module to be imported into other scripts (e.g., training, inference).

Functions:
- undistort_image(image): Removes camera lens distortion.
- apply_white_balance(image): Corrects color cast using Gray World algorithm.
- apply_clahe(image): Enhances local contrast.
- apply_gamma_correction(image): Adjusts non-linear brightness (good for shadows).
- preprocess_image(image): Runs the full pipeline on a single image.
"""

import cv2
import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm

# Import from unified config system
try:
    from ..config import config
except ImportError:
    # Fallback: try importing from preprocessing config
    try:
        from . import config as preprocessing_config
        # Create a simple config object for backward compatibility
        class SimpleConfig:
            CameraMat = preprocessing_config.CameraMat
            DistCoeff = preprocessing_config.DistCoeff
            UNDISTORT_ALPHA = preprocessing_config.UNDISTORT_ALPHA
            CLAHE_CLIP_LIMIT = preprocessing_config.CLAHE_CLIP_LIMIT
            CLAHE_TILE_GRID_SIZE = preprocessing_config.CLAHE_TILE_GRID_SIZE
            GAMMA = preprocessing_config.GAMMA
        config = SimpleConfig()
    except ImportError:
        print("ERROR: config not found!")
        raise


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray = None,
                   dist_coeffs: np.ndarray = None, alpha: float = None) -> tuple:
    """
    Removes lens distortion from an image using camera parameters.
    
    Args:
        image: The input image (NumPy array).
        camera_matrix: 3x3 camera matrix (if None, uses config)
        dist_coeffs: Distortion coefficients (if None, uses config)
        alpha: Undistort alpha parameter (if None, uses config)
        
    Returns:
        Tuple of (undistorted_image, new_camera_matrix)
    """
    h, w = image.shape[:2]
    
    # Get parameters from config if not provided
    if camera_matrix is None:
        try:
            from ..config import config as unified_config
            camera_matrix = unified_config.get_camera_matrix(w, h)
        except:
            camera_matrix = config.CameraMat
    
    if dist_coeffs is None:
        try:
            from ..config import config as unified_config
            dist_coeffs = unified_config.CAMERA['dist_coeffs']
        except:
            dist_coeffs = config.DistCoeff
    
    if alpha is None:
        try:
            from ..config import config as unified_config
            alpha = unified_config.PREPROCESSING['undistort_alpha']
        except:
            alpha = config.UNDISTORT_ALPHA
    
    # Get the new optimal camera matrix based on the ALPHA
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, 
        dist_coeffs, 
        (w, h), 
        alpha, 
        (w, h)
    )
    
    # Undistort the image
    undistorted = cv2.undistort(
        image, 
        camera_matrix, 
        dist_coeffs, 
        None, 
        new_camera_matrix
    )
    
    # Crop the image based on the ROI if alpha=0
    # Note: With alpha=0, roi contains the valid pixel area
    if alpha == 0:
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        
    return undistorted, new_camera_matrix


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Applies the "Gray World" white balance algorithm.
    Assumes the average color of the entire scene is gray.
    
    Args:
        image: The input image (NumPy array).
        
    Returns:
        The white-balanced image.
    """
    # Split the image into its B, G, R channels
    b, g, r = cv2.split(image)
    
    # Calculate the average of each channel
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # Calculate the overall average
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # Calculate the scaling factor for each channel
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    # Scale the channels
    b_balanced = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g_balanced = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r_balanced = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    # Merge the balanced channels back together
    return cv2.merge([b_balanced, g_balanced, r_balanced])


def apply_clahe(image: np.ndarray, clip_limit: float = None,
                tile_grid_size: tuple = None) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance local contrast.
    
    Args:
        image: The input image (NumPy array).
        clip_limit: CLAHE clip limit (if None, uses config)
        tile_grid_size: CLAHE tile grid size (if None, uses config)
        
    Returns:
        The contrast-enhanced image.
    """
    # Get parameters from config if not provided
    if clip_limit is None:
        try:
            from ..config import config as unified_config
            clip_limit = unified_config.PREPROCESSING['clahe_clip_limit']
        except:
            clip_limit = config.CLAHE_CLIP_LIMIT
    
    if tile_grid_size is None:
        try:
            from ..config import config as unified_config
            tile_grid_size = unified_config.PREPROCESSING['clahe_tile_grid_size']
        except:
            tile_grid_size = config.CLAHE_TILE_GRID_SIZE
    
    # Convert to LAB color space to apply CLAHE only to the Lightness channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create the CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    
    # Apply CLAHE to the L-channel
    l_clahe = clahe.apply(l)
    
    # Merge the channels back and convert to BGR
    lab_enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Applies non-linear gamma correction to the image.
    
    This is superior to linear brightness/contrast as it brightens
    shadows and mid-tones more than highlights, preventing blowouts.
    
    Args:
        image: The input image (NumPy array).
        gamma: Gamma value (if None, uses config)
        
    Returns:
        The gamma-corrected image.
    """
    # Get gamma from config if not provided
    if gamma is None:
        try:
            from ..config import config as unified_config
            gamma = unified_config.PREPROCESSING['gamma']
        except:
            gamma = config.GAMMA
    
    # Build a lookup table (LUT) mapping pixel values [0, 255]
    # to their new gamma-corrected values.
    # Uses gamma, but applies the inverse formula.
    inv_gamma = 1.0 / gamma
    
    # This creates a 256-element array
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    
    # Apply the LUT to every pixel in the image (very fast)
    return cv2.LUT(image, lut)


def preprocess_image(image: np.ndarray, camera_matrix: np.ndarray = None,
                    dist_coeffs: np.ndarray = None, return_camera_matrix: bool = False) -> tuple:
    """
    Runs the full preprocessing pipeline on a single image.
    This is the function you'll likely import into your training/inference scripts.
    
    Args:
        image: The raw input image (NumPy array, RGB or BGR).
        camera_matrix: Optional camera matrix (if None, uses config)
        dist_coeffs: Optional distortion coefficients (if None, uses config)
        return_camera_matrix: If True, returns (processed_image, new_camera_matrix)
        
    Returns:
        Preprocessed image, or (preprocessed_image, new_camera_matrix) if return_camera_matrix=True
    """
    # Step 1: Fix geometric distortion
    processed, new_camera_matrix = undistort_image(image, camera_matrix, dist_coeffs)
    
    # Step 2: Fix color cast
    processed = apply_white_balance(processed)
    
    # Step 3: Enhance local contrast
    processed = apply_clahe(processed)
    
    # Step 4: Final non-linear brightness adjustment
    processed = apply_gamma_correction(processed)
    
    if return_camera_matrix:
        return processed, new_camera_matrix
    return processed


# --- Standalone Script Execution ---

if __name__ == "__main__":
    """
    This block runs when you execute the script directly from the terminal.
    It processes an entire folder of images.
    """
    parser = argparse.ArgumentParser(
        description="Batch preprocess images from a folder."
    )
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Path to the folder containing raw images."
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Path to the folder where processed images will be saved."
    )
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        exit(1)
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting batch preprocessing...")
    print(f"Input folder:  {args.input_dir}")
    print(f"Output folder: {args.output_dir}")
    
    # Find all images
    image_paths = glob(os.path.join(args.input_dir, '*.jpg')) + \
                  glob(os.path.join(args.input_dir, '*.jpeg')) + \
                  glob(os.path.join(args.input_dir, '*.png'))
                  
    if not image_paths:
        print("Error: No .jpg, .jpeg, or .png images found.")
        exit(1)
        
    print(f"Found {len(image_paths)} images to process.")
    
    # Process each image with a progress bar
    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # 1. Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"\nWarning: Failed to read {img_path}. Skipping.")
                continue
                
            # 2. Run the pipeline
            processed_image = preprocess_image(image)
            
            # 3. Save the result
            filename = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, filename)
            cv2.imwrite(output_path, processed_image)
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}. Skipping.")
            
    print("\nBatch processing complete.")
    print(f"Processed images saved to: {args.output_dir}")



