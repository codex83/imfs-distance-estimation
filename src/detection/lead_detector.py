"""
Hybrid Lead Vehicle Detection

This module uses YOLOv8 for vehicle detection and a combined filtering
(fixed polygon) and scoring (weighted heuristics) approach to find the
lead vehicle.

It is based on 'lead_vehicle_detector.py' but removes all classic
computer vision lane detection in favor of a fixed ROI filter.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import csv
from typing import Dict, List, Optional, Tuple


class HybridLeadDetector:
    def __init__(self, model_path='yolov8m.pt', hood_exclude_ratio=0.20, 
                 conf_threshold=0.25, device=None):
        """
        Initialize the hybrid lead vehicle detector.
        
        Args:
            model_path: Path to YOLO model (default: yolov8m.pt)
            hood_exclude_ratio: Bottom portion of image to exclude (0.20 = 20%)
            conf_threshold: YOLO confidence threshold (default: 0.25)
            device: Device to run inference on. None = auto-detect (cuda > mps > cpu)
        """
        # Auto-detect best available device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("Using CUDA (NVIDIA GPU)")
            elif torch.backends.mps.is_available():
                device = 'mps'
                print("Using MPS (Apple Silicon)")
            else:
                device = 'cpu'
                print("Using CPU")
        else:
            print(f"Using specified device: {device}")
        
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        
        self.hood_exclude_ratio = hood_exclude_ratio
        self.conf_threshold = conf_threshold
        
        # Fixed ROI polygon, defined by ratios (from File 2 logic)
        # This polygon is created dynamically based on image size
        self.roi_poly_ratios = np.array([
            [0.1, 1.0],     # Bottom-left (10% width, 100% height)
            [0.45, 0.5],    # Top-left (45% width, 50% height)
            [0.55, 0.5],    # Top-right (55% width, 50% height)
            [0.90, 1.0]     # Bottom-right (90% width, 100% height)
        ], dtype=np.float32)

    def get_fixed_roi(self, img_height, img_width):
        """Creates the fixed ROI polygon based on image dimensions"""
        poly_points = self.roi_poly_ratios.copy()
        poly_points[:, 0] *= img_width
        poly_points[:, 1] *= img_height
        return poly_points.astype(np.int32)

    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return np.array([cx, cy], dtype=np.float32)

    # --- Scoring Functions (from File 1) ---
    
    def calculate_closeness_score(self, bbox, img_height):
        """Calculate how close the vehicle is based on vertical position"""
        cy = (bbox[1] + bbox[3]) / 2
        norm_cy = cy / img_height
        if norm_cy > 0.8:
            return 1.0
        elif norm_cy > 0.5:
            return 0.5 + (norm_cy - 0.5) / 0.3 * 0.5
        else:
            return norm_cy
    
    def calculate_size_score(self, bbox, img_width, img_height):
        """Calculate size score - larger vehicles are typically closer"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        frame_area = img_width * img_height
        relative_size = area / frame_area
        if relative_size > 0.4:
            return 0.2
        elif relative_size > 0.03:
            return min(relative_size / 0.15, 1.0)
        else:
            return relative_size / 0.03 * 0.5
    
    def calculate_center_alignment_score(self, bbox, img_width):
        """Calculate how aligned the vehicle is with image center"""
        cx = (bbox[0] + bbox[2]) / 2
        center_deviation = abs(cx - img_width / 2) / (img_width / 2)
        if center_deviation < 0.15:
            return 1.0
        elif center_deviation < 0.3:
            return 1.0 - (center_deviation - 0.15) / 0.15 * 0.5
        else:
            return 0.5 - min((center_deviation - 0.3) / 0.3, 0.4)

    def find_lead_vehicle(self, image_path, visualize=False):
        """
        Find the lead vehicle in a single image using fixed filters and weighted scoring.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"ERROR: Could not read image: {image_path}")
                return None, None
            
            h, w = img.shape[:2]
            filename = os.path.basename(image_path)
            
            # Extract ground truth distance from filename (from File 1)
            ground_truth_distance = None
            try:
                if '_dist' in filename:
                    dist_part = filename.split('_dist')[1].split('_')[0]
                    ground_truth_distance = float(dist_part)
            except:
                pass
            
            # --- Detection ---
            results = self.model(img, verbose=False, 
                               conf=self.conf_threshold, device=self.device)[0]
            
            # Filter for vehicle classes
            vehicle_classes = [2, 5, 7, 3] # car, bus, truck, motorcycle
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls in vehicle_classes:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append({
                            'bbox': bbox,
                            'conf': conf,
                            'class': cls
                        })
            
            # --- Filtering (from File 2 Logic) ---
            
            # 1. Create fixed polygon filter
            roi_polygon = self.get_fixed_roi(h, w)
            polygon_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [roi_polygon], 255)
            
            # 2. Create hood exclusion filter
            hood_cutoff_y = h * (1.0 - self.hood_exclude_ratio)
            
            candidate_detections = []
            all_vehicles_data = [] # For JSON output

            for det in detections:
                x1, y1, x2, y2 = det['bbox'].astype(int)
                
                # Store all vehicle data for JSON
                all_vehicles_data.append({
                    'bbox': det['bbox'].tolist(),
                    'confidence': float(det['conf']),
                    'in_roi': False # Will be updated if it passes
                })

                # Filter 1: Hood Exclusion (from File 2)
                if y1 > hood_cutoff_y:
                    continue
                
                # Filter 2: Polygon Overlap (from File 2)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area == 0:
                    continue
                
                # Ensure coordinates are within image bounds for slicing
                y1_c, y2_c = max(0, y1), min(h, y2)
                x1_c, x2_c = max(0, x1), min(w, x2)
                
                intersection_area = cv2.countNonZero(polygon_mask[y1_c:y2_c, x1_c:x2_c])
                overlap_ratio = intersection_area / box_area
                
                if overlap_ratio < 0.3:
                    continue
                
                # This is a valid candidate
                candidate_detections.append(det)
                # Update the 'in_roi' status for the output JSON
                all_vehicles_data[-1]['in_roi'] = True

            # --- Scoring (from File 1 Logic) ---
            lead_detection = None
            if len(candidate_detections) > 0:
                best_score = -1
                best_idx = -1
                
                for idx, det in enumerate(candidate_detections):
                    bbox = det['bbox']
                    # Apply File 1's weighted scoring
                    closeness = self.calculate_closeness_score(bbox, h)
                    size = self.calculate_size_score(bbox, w, h)
                    alignment = self.calculate_center_alignment_score(bbox, w)
                    
                    score = closeness * 0.7 + size * 0.2 + alignment * 0.1
                    
                    det['score'] = score # Store score for visualization
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx >= 0:
                    lead_detection = candidate_detections[best_idx]
            
            # --- Build Result ---
            lead_vehicle = None
            if lead_detection is not None:
                lead_vehicle = {
                    'bbox': lead_detection['bbox'].tolist(),
                    'confidence': float(lead_detection['conf']),
                    'in_roi': True,
                    'is_lead': True,
                    'score': float(lead_detection['score'])
                }

            metadata = {
                'total_vehicles': len(all_vehicles_data),
                'roi_vehicles': len(candidate_detections),
                'image_dimensions': {'width': w, 'height': h}
            }
            
            result = {
                'filename': filename,
                'ground_truth_distance': ground_truth_distance,
                'all_vehicles': all_vehicles_data,
                'lead_vehicle': lead_vehicle,
                'metadata': metadata
            }
            
            annotated_img = None
            if visualize:
                annotated_img = self._create_visualization(
                    img, roi_polygon, all_vehicles_data,
                    lead_vehicle, hood_cutoff_y
                )
            
            return result, annotated_img
            
        except Exception as e:
            print(f"ERROR processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def _create_visualization(self, img, roi_polygon, all_vehicles,
                              lead_vehicle, hood_cutoff_y):
        """Create visualization with fixed ROI and detections"""
        annotated_img = img.copy()
        h, w = img.shape[:2]
        
        # Draw ROI
        cv2.polylines(annotated_img, [roi_polygon], True, (0, 255, 255), 3)
        
        # Draw hood exclusion
        cv2.line(annotated_img, (0, int(hood_cutoff_y)), (w, int(hood_cutoff_y)), (255, 0, 255), 2)
        cv2.putText(annotated_img, "Hood Filter", (10, int(hood_cutoff_y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Draw all detections
        for det in all_vehicles:
            if det['in_roi']:
                color = (255, 255, 0) # Yellow for ROI
            else:
                color = (128, 128, 128) # Gray for non-ROI
                
            bbox = np.array(det['bbox']).astype(int)
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), color, 2)
        
        # Draw lead vehicle
        if lead_vehicle is not None:
            bbox = np.array(lead_vehicle['bbox']).astype(int)
            color = (0, 255, 0) # Green for Lead
            
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, 4)
            
            label = f"LEAD (Score: {lead_vehicle['score']:.2f})"
            cv2.putText(annotated_img, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return annotated_img


# --- Main Processing Function (from File 1) ---

def process_dashcam_images(
    input_dir: str,
    output_json: str,
    save_visualizations: bool = False,
    output_viz_dir: Optional[str] = None,
    save_csv: bool = False,
    output_csv: Optional[str] = None,
    model_path: str = 'yolov8m.pt',
    hood_exclude_ratio: float = 0.20,
    conf_threshold: float = 0.25,
    device: Optional[str] = None
) -> Dict:
    """
    Process dashcam images to detect lead vehicles and save results.
    
    Args:
        input_dir: Directory containing dashcam images
        output_json: Path to save JSON results
        save_visualizations: Whether to save annotated images
        output_viz_dir: Directory for visualizations
        save_csv: Whether to save CSV results
        output_csv: Path for CSV file
        model_path: Path to YOLO model weights
        hood_exclude_ratio: Bottom portion of image to exclude (car hood)
        conf_threshold: YOLO confidence threshold
        device: Device for inference ('cuda', 'mps', 'cpu', or None for auto)
    
    Returns:
        dict: Summary statistics.
    """
    # Validate inputs
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if save_visualizations and output_viz_dir is None:
        raise ValueError("output_viz_dir must be specified when save_visualizations=True")
    
    # Create output directories
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    if save_visualizations:
        os.makedirs(output_viz_dir, exist_ok=True)
    
    # Initialize detector
    print(f"\n{'='*60}")
    print("Hybrid Lead Vehicle Detector - Processing Started")
    print(f"{'='*60}")
    detector = HybridLeadDetector(
        model_path=model_path,
        hood_exclude_ratio=hood_exclude_ratio,
        conf_threshold=conf_threshold,
        device=device
    )
    
    # Get all images
    image_paths = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    print(f"\nFound {len(image_paths)} images in '{input_dir}'")
    print(f"Configuration:")
    print(f"  - Model: {model_path}")
    print(f"  - Confidence Threshold: {conf_threshold}")
    print(f"  - Hood Exclusion: {hood_exclude_ratio*100:.0f}%")
    print(f"  - Save Visualizations: {save_visualizations}")
    print(f"  - Save CSV: {save_csv}")
    print(f"\nProcessing...")
    
    all_results = {}
    csv_rows = []
    
    for i, img_path in enumerate(image_paths):
        result, annotated = detector.find_lead_vehicle(
            img_path, 
            visualize=save_visualizations
        )
        
        if result is not None:
            all_results[result['filename']] = {
                'ground_truth_distance': result['ground_truth_distance'],
                'all_vehicles': result['all_vehicles'],
                'lead_vehicle': result['lead_vehicle'],
                'metadata': result['metadata']
            }
            
            if save_csv:
                lead = result['lead_vehicle']
                if lead is not None:
                    csv_rows.append({
                        'filename': result['filename'],
                        'ground_truth_distance': result['ground_truth_distance'] if result['ground_truth_distance'] is not None else '',
                        'lead_x1': lead['bbox'][0],
                        'lead_y1': lead['bbox'][1],
                        'lead_x2': lead['bbox'][2],
                        'lead_y2': lead['bbox'][3],
                        'lead_confidence': lead['confidence'],
                        'total_vehicles': result['metadata']['total_vehicles']
                    })
                else:
                    csv_rows.append({
                        'filename': result['filename'],
                        'ground_truth_distance': result['ground_truth_distance'] if result['ground_truth_distance'] is not None else '',
                        'lead_x1': '', 'lead_y1': '', 'lead_x2': '', 'lead_y2': '',
                        'lead_confidence': '',
                        'total_vehicles': result['metadata']['total_vehicles']
                    })
            
            if save_visualizations and annotated is not None:
                out_path = os.path.join(output_viz_dir, result['filename'])
                cv2.imwrite(out_path, annotated)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  Progress: {i + 1}/{len(image_paths)} images ({(i+1)/len(image_paths)*100:.1f}%)")
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ JSON saved to: {output_json}")
    
    # Save CSV
    csv_path = None
    if save_csv:
        csv_path = output_csv or output_json.replace('.json', '.csv')
        with open(csv_path, 'w', newline='') as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
        print(f"✓ CSV saved to: {csv_path}")
    
    if save_visualizations:
        print(f"✓ Visualizations saved to: {output_viz_dir}")
    
    # Calculate summary
    frames_with_lead = sum(1 for r in all_results.values() if r['lead_vehicle'] is not None)
    total_frames = len(all_results)
    detection_rate = (frames_with_lead / total_frames * 100) if total_frames > 0 else 0
    
    summary = {
        'total_frames': total_frames,
        'frames_with_lead': frames_with_lead,
        'detection_rate': detection_rate,
        'output_files': { 'json': output_json, 'csv': csv_path, 'visualizations': output_viz_dir }
    }
    
    print(f"\n{'='*60}")
    print("Detection Summary")
    print(f"{'='*60}")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with lead vehicle: {frames_with_lead}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"{'='*60}\n")
    
    return summary



