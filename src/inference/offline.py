"""
Offline Inference

Offline inference for pre-recorded videos and images using either heavyweight or lightweight model.
"""

import cv2
import os
import json
from typing import Optional, List, Dict
from pathlib import Path
from .model_runner import ModelRunner


class OfflineInference:
    """
    Offline inference for pre-recorded videos and images.
    
    Can use either heavyweight or lightweight model.
    """
    
    def __init__(self, model_type: str = 'lightweight', checkpoint_path: Optional[str] = None,
                 heavyweight_checkpoint_path: Optional[str] = None,
                 lightweight_checkpoint_path: Optional[str] = None,
                 detection_model_path: str = 'yolov8m.pt', device: Optional[str] = None):
        """
        Initialize offline inference.
        
        Args:
            model_type: 'heavyweight', 'lightweight', or 'both'
            checkpoint_path: Path to model checkpoint (if using single model)
            heavyweight_checkpoint_path: Path to heavyweight checkpoint (if using both)
            lightweight_checkpoint_path: Path to lightweight checkpoint (if using both)
            detection_model_path: Path to YOLO detection model
            device: Device to run on (None for auto-detect)
        """
        self.model_type = model_type
        
        if model_type == 'both':
            if not heavyweight_checkpoint_path or not lightweight_checkpoint_path:
                raise ValueError("Both heavyweight_checkpoint_path and lightweight_checkpoint_path required for model_type='both'")
            self.heavyweight_runner = ModelRunner(
                model_type='heavyweight',
                checkpoint_path=heavyweight_checkpoint_path,
                detection_model_path=detection_model_path,
                device=device
            )
            self.lightweight_runner = ModelRunner(
                model_type='lightweight',
                checkpoint_path=lightweight_checkpoint_path,
                detection_model_path=detection_model_path,
                device=device
            )
        else:
            checkpoint = checkpoint_path or (heavyweight_checkpoint_path if model_type == 'heavyweight' else lightweight_checkpoint_path)
            if not checkpoint:
                raise ValueError(f"checkpoint_path required for model_type='{model_type}'")
            
            self.runner = ModelRunner(
                model_type=model_type,
                checkpoint_path=checkpoint,
                detection_model_path=detection_model_path,
                device=device
            )
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save annotated image
            
        Returns:
            Dictionary with predictions and metadata
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        results = {}
        
        if self.model_type == 'both':
            heavyweight_dist, heavyweight_meta = self.heavyweight_runner.predict(image)
            lightweight_dist, lightweight_meta = self.lightweight_runner.predict(image)
            
            results = {
                'image_path': image_path,
                'heavyweight': {
                    'distance': heavyweight_dist,
                    'metadata': heavyweight_meta
                },
                'lightweight': {
                    'distance': lightweight_dist,
                    'metadata': lightweight_meta
                }
            }
            
            # Draw both predictions
            if output_path:
                annotated = image.copy()
                if heavyweight_dist is not None:
                    cv2.putText(annotated, f"Heavyweight: {heavyweight_dist:.2f}m", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if lightweight_dist is not None:
                    cv2.putText(annotated, f"Lightweight: {lightweight_dist:.2f}m", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(output_path, annotated)
        else:
            distance, metadata = self.runner.predict(image)
            results = {
                'image_path': image_path,
                'distance': distance,
                'metadata': metadata
            }
            
            # Draw prediction
            if output_path:
                annotated = image.copy()
                if distance is not None:
                    cv2.putText(annotated, f"Distance: {distance:.2f}m", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if 'bbox' in metadata:
                        x1, y1, x2, y2 = map(int, metadata['bbox'])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(output_path, annotated)
        
        return results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                      save_json: bool = False, json_path: Optional[str] = None) -> Dict:
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            save_json: Whether to save frame-by-frame results to JSON
            json_path: Path to save JSON results
            
        Returns:
            Dictionary with summary statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_num = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if self.model_type == 'both':
                    heavyweight_dist, heavyweight_meta = self.heavyweight_runner.predict(frame)
                    lightweight_dist, lightweight_meta = self.lightweight_runner.predict(frame)
                    
                    frame_result = {
                        'frame': frame_num,
                        'heavyweight': {
                            'distance': heavyweight_dist,
                            'metadata': heavyweight_meta
                        },
                        'lightweight': {
                            'distance': lightweight_dist,
                            'metadata': lightweight_meta
                        }
                    }
                    
                    # Draw both predictions
                    if writer:
                        annotated = frame.copy()
                        if heavyweight_dist is not None:
                            cv2.putText(annotated, f"Heavyweight: {heavyweight_dist:.2f}m", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if lightweight_dist is not None:
                            cv2.putText(annotated, f"Lightweight: {lightweight_dist:.2f}m", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        writer.write(annotated)
                else:
                    distance, metadata = self.runner.predict(frame)
                    frame_result = {
                        'frame': frame_num,
                        'distance': distance,
                        'metadata': metadata
                    }
                    
                    # Draw prediction
                    if writer:
                        annotated = frame.copy()
                        if distance is not None:
                            cv2.putText(annotated, f"Distance: {distance:.2f}m", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if 'bbox' in metadata:
                                x1, y1, x2, y2 = map(int, metadata['bbox'])
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        writer.write(annotated)
                
                frame_results.append(frame_result)
                frame_num += 1
                
                if frame_num % 30 == 0:
                    print(f"Processed {frame_num}/{total_frames} frames...")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Save JSON if requested
        if save_json:
            json_output_path = json_path or video_path.replace('.mp4', '_results.json')
            with open(json_output_path, 'w') as f:
                json.dump(frame_results, f, indent=2)
            print(f"Saved results to {json_output_path}")
        
        # Calculate summary statistics
        summary = {
            'total_frames': frame_num,
            'video_path': video_path
        }
        
        if self.model_type == 'both':
            heavyweight_distances = [r['heavyweight']['distance'] for r in frame_results if r['heavyweight']['distance'] is not None]
            lightweight_distances = [r['lightweight']['distance'] for r in frame_results if r['lightweight']['distance'] is not None]
            
            summary['heavyweight'] = {
                'frames_with_prediction': len(heavyweight_distances),
                'avg_distance': sum(heavyweight_distances) / len(heavyweight_distances) if heavyweight_distances else None
            }
            summary['lightweight'] = {
                'frames_with_prediction': len(lightweight_distances),
                'avg_distance': sum(lightweight_distances) / len(lightweight_distances) if lightweight_distances else None
            }
        else:
            distances = [r['distance'] for r in frame_results if r['distance'] is not None]
            summary['frames_with_prediction'] = len(distances)
            summary['avg_distance'] = sum(distances) / len(distances) if distances else None
        
        print(f"\nProcessed {frame_num} frames total.")
        return summary
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None,
                         extensions: List[str] = None) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Optional directory to save annotated images
            extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            List of result dictionaries
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png']
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(input_dir).glob(f'*{ext}'))
            image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        results = []
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, image_path.name)
            
            try:
                result = self.process_image(str(image_path), output_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
        
        print(f"Processed {len(image_paths)} images total.")
        return results



