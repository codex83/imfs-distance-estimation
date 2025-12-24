"""
Model Runner

Unified interface for loading and running distance estimation models.
Updated to match Inference_full.ipynb approach: detection on preprocessed images.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict
from ultralytics import YOLO
from ..models import HeavyweightModel, LightweightModel
from ..preprocessing import preprocess_image
from ..config import config


class ModelRunner:
    """
    Unified interface for running distance estimation models.
    
    Handles model loading, preprocessing, detection (on preprocessed images), and inference.
    """
    
    def __init__(self, model_type='lightweight', checkpoint_path=None, 
                 detection_model_path='yolov8m.pt', device=None, precision='float32'):
        """
        Initialize the model runner.
        
        Args:
            model_type: 'heavyweight' or 'lightweight'
            checkpoint_path: Path to model checkpoint
            detection_model_path: Path to YOLO detection model
            device: Device to run on (None for auto-detect)
            precision: 'float32' or 'float16' for inference
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        self.device = device
        self.model_type = model_type
        self.precision = precision
        
        # Load distance estimation model
        if model_type == 'heavyweight':
            self.model = HeavyweightModel().to(device)
        elif model_type == 'lightweight':
            self.model = LightweightModel().to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'heavyweight' or 'lightweight'")
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Convert to float16 if requested
        if precision == 'float16' and device.type == 'cuda':
            self.model = self.model.half()
        
        # Initialize YOLO model for detection
        self.yolo_model = YOLO(detection_model_path)
        self.yolo_model.to(str(device))
        
        # Detection configuration
        self.conf_threshold = config.DETECTION['confidence_threshold']
        self.hood_exclude_ratio = config.DETECTION['hood_exclude_ratio']
        
        # Image sizes (matching training)
        self.full_img_size = config.INFERENCE['full_img_size']
        self.patch_size = config.INFERENCE['patch_size']
        
        # Normalization (ImageNet)
        self.mean = np.array(config.INFERENCE['normalize_mean'])
        self.std = np.array(config.INFERENCE['normalize_std'])
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint with support for multiple formats."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Direct state dict
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {self.model_type} model from {checkpoint_path}")
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    def preprocess_frame(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess a single frame/image and return updated camera parameters.
        
        This is the wrapper function that handles camera matrix updates.
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            Tuple of (preprocessed_image, fx_new, fy_new)
        """
        h, w = image.shape[:2]
        
        # Build camera matrix from config
        camera_matrix = config.get_camera_matrix(w, h)
        dist_coeffs = config.CAMERA['dist_coeffs']
        
        # Convert BGR to RGB for preprocessing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess and get updated camera matrix
        processed, new_camera_matrix = preprocess_image(
            image_rgb,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            return_camera_matrix=True
        )
        
        # Extract updated focal lengths
        fx_new = new_camera_matrix[0, 0]
        fy_new = new_camera_matrix[1, 1]
        
        return processed, fx_new, fy_new
    
    def extract_geometric_features(self, bbox: np.ndarray, img_width: int, img_height: int,
                                  camera_matrix: np.ndarray) -> np.ndarray:
        """
        Extract 10 geometric features from bbox and camera matrix.
        
        Uses the notebook's approach with dynamic camera matrix.
        
        Args:
            bbox: [x1, y1, x2, y2]
            img_width, img_height: Image dimensions (after preprocessing)
            camera_matrix: 3x3 camera matrix
            
        Returns:
            10 features normalized to [0, 1]
        """
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        
        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1
        bbox_center_x = (bbox_x1 + bbox_x2) / 2
        bbox_center_y = (bbox_y1 + bbox_y2) / 2
        bbox_aspect_ratio = bbox_height / (bbox_width + 1e-6)
        
        feature_1 = bbox_width / img_width
        feature_2 = bbox_height / img_height
        feature_3 = bbox_y1 / img_height
        feature_4 = bbox_y2 / img_height
        feature_5 = bbox_center_x / img_width
        feature_6 = bbox_center_y / img_height
        feature_7 = bbox_aspect_ratio
        
        # Features 8-10: Calibrated features using camera matrix
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cy = camera_matrix[1, 2]
        AVG_CAR_HEIGHT = config.CAMERA['avg_car_height']
        
        distance_estimate = (AVG_CAR_HEIGHT * fy) / (bbox_height + 1e-6)
        distance_estimate_norm = np.clip(distance_estimate / 100.0, 0, 1)
        
        vertical_angle = np.arctan2(bbox_y2 - cy, fy)
        vertical_angle_norm = (vertical_angle + np.pi/2) / np.pi
        
        angular_height = 2 * np.arctan(bbox_height / (2 * fy))
        angular_height_norm = angular_height / (np.pi/2)
        
        features = np.array([
            feature_1, feature_2, feature_3, feature_4, feature_5,
            feature_6, feature_7, distance_estimate_norm,
            vertical_angle_norm, angular_height_norm
        ], dtype=np.float32)
        
        return features
    
    def extract_all_features(self, preprocessed_image: np.ndarray, bbox: np.ndarray,
                            fx: float, fy: float) -> Dict[str, torch.Tensor]:
        """
        Extract all 3-branch features (notebook's approach).
        
        Args:
            preprocessed_image: Preprocessed RGB image
            bbox: Bounding box [x1, y1, x2, y2]
            fx, fy: Updated focal lengths after preprocessing
            
        Returns:
            Dictionary with 'full_image', 'car_patch', 'geometric' tensors
        """
        h, w = preprocessed_image.shape[:2]
        
        # Build camera matrix with updated focal lengths
        camera_matrix = np.array([
            [fx, 0, w/2],
            [0, fy, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Branch 1: Full image
        full_img = cv2.resize(preprocessed_image, self.full_img_size)
        full_img = full_img.astype(np.float32) / 255.0
        full_img = torch.from_numpy(full_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize
        mean_tensor = torch.tensor(self.mean).view(3, 1, 1)
        std_tensor = torch.tensor(self.std).view(3, 1, 1)
        full_img = (full_img - mean_tensor) / std_tensor
        
        # Branch 2: Vehicle patch
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        patch = preprocessed_image[y1:y2, x1:x2]
        patch = cv2.resize(patch, self.patch_size)
        patch = patch.astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize
        patch = (patch - mean_tensor) / std_tensor
        
        # Branch 3: Geometric features
        geometric = self.extract_geometric_features(bbox, w, h, camera_matrix)
        geometric = torch.from_numpy(geometric).float().unsqueeze(0)  # (1, 10)
        
        return {
            'full_image': full_img,
            'car_patch': patch,
            'geometric': geometric
        }
    
    def _detect_lead_vehicle_on_preprocessed(self, preprocessed_image: np.ndarray) -> Optional[list]:
        """
        Detect lead vehicle on PREPROCESSED image (matching notebook approach).
        
        Args:
            preprocessed_image: Preprocessed RGB image
            
        Returns:
            Bounding box [x1, y1, x2, y2] or None
        """
        h, w = preprocessed_image.shape[:2]
        
        # Convert RGB to BGR for YOLO (YOLO expects BGR)
        image_bgr = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection on preprocessed image
        results = self.yolo_model(image_bgr, verbose=False,
                                 conf=self.conf_threshold,
                                 device=str(self.device))[0]
        
        # Filter for vehicle classes
        vehicle_classes = [2, 5, 7, 3]  # car, bus, truck, motorcycle
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Hood filtering on preprocessed image
                    hood_threshold = h * (1 - self.hood_exclude_ratio)
                    if bbox[3] < hood_threshold:  # y2 < hood_threshold
                        detections.append({'bbox': bbox, 'conf': conf})
        
        if not detections:
            return None
        
        # Score and select best detection
        best_score = -1
        best_idx = -1
        
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            
            # Calculate scores (same as notebook)
            cy = (bbox[1] + bbox[3]) / 2
            norm_cy = cy / h
            if norm_cy > 0.8:
                closeness = 1.0
            elif norm_cy > 0.5:
                closeness = 0.5 + (norm_cy - 0.5) / 0.3 * 0.5
            else:
                closeness = norm_cy
            
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            relative_size = area / (w * h)
            if relative_size > 0.4:
                size = 0.2
            elif relative_size > 0.03:
                size = min(relative_size / 0.15, 1.0)
            else:
                size = relative_size / 0.03 * 0.5
            
            cx = (bbox[0] + bbox[2]) / 2
            center_deviation = abs(cx - w / 2) / (w / 2)
            if center_deviation < 0.15:
                alignment = 1.0
            elif center_deviation < 0.3:
                alignment = 1.0 - (center_deviation - 0.15) / 0.15 * 0.5
            else:
                alignment = 0.5 - min((center_deviation - 0.3) / 0.3, 0.4)
            
            score = closeness * 0.7 + size * 0.2 + alignment * 0.1
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx < 0:
            return None
        
        return detections[best_idx]['bbox'].tolist()
    
    def predict(self, image: np.ndarray, bbox: Optional[list] = None,
                save_stages: bool = False, stage_output_dir: Optional[str] = None) -> Tuple[float, Dict]:
        """
        Predict distance for a single image.
        
        Matches notebook approach: preprocessing -> detection on preprocessed -> feature extraction -> inference.
        
        Args:
            image: Input image (BGR format from cv2)
            bbox: Optional bounding box [x1, y1, x2, y2]. If None, will detect on preprocessed image.
            save_stages: If True, save intermediate stage images
            stage_output_dir: Directory to save stage images (required if save_stages=True)
            
        Returns:
            Tuple of (distance_meters, metadata_dict)
        """
        # Stage 1: Preprocess image
        preprocessed, fx_new, fy_new = self.preprocess_frame(image)
        
        if save_stages and stage_output_dir:
            os.makedirs(stage_output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(stage_output_dir, '01_preprocessing.jpg'),
                       cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
        
        # Stage 2: Detect lead vehicle on PREPROCESSED image
        if bbox is None:
            bbox = self._detect_lead_vehicle_on_preprocessed(preprocessed)
            if bbox is None:
                return None, {'error': 'No lead vehicle detected'}
        
        if save_stages and stage_output_dir:
            annotated_preprocessed = preprocessed.copy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_preprocessed, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.imwrite(os.path.join(stage_output_dir, '02_detection_bbox.jpg'),
                       cv2.cvtColor(annotated_preprocessed, cv2.COLOR_RGB2BGR))
        
        # Stage 3: Extract features (using notebook's approach)
        features = self.extract_all_features(preprocessed, bbox, fx_new, fy_new)
        
        # Move to device
        full_img = features['full_image'].to(self.device)
        car_patch = features['car_patch'].to(self.device)
        geometric = features['geometric'].to(self.device)
        
        # Convert to float16 if needed
        if self.precision == 'float16' and self.device.type == 'cuda':
            full_img = full_img.half()
            car_patch = car_patch.half()
            geometric = geometric.half()
        
        # Stage 4: Run inference
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type):
                distance = self.model(full_img, car_patch, geometric)
        
        distance_meters = distance.cpu().item()
        
        metadata = {
            'bbox': bbox,
            'model_type': self.model_type,
            'fx': fx_new,
            'fy': fy_new
        }
        
        if save_stages and stage_output_dir:
            # Stage 5: Final output visualization
            annotated_final = image.copy()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_final, (x1, y1), (x2, y2), (0, 255, 0), 4)
            label = f"{distance_meters:.2f}m"
            cv2.putText(annotated_final, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(stage_output_dir, '03_final_output.jpg'), annotated_final)
        
        return distance_meters, metadata
