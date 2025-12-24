"""
Real-time Inference

Real-time video inference using the lightweight model for live camera feeds.
"""

import cv2
import numpy as np
from typing import Optional, Callable
from .model_runner import ModelRunner


class RealtimeInference:
    """
    Real-time inference for video streams.
    
    Uses the lightweight model for fast inference on live camera feeds.
    """
    
    def __init__(self, checkpoint_path: str, detection_model_path: str = 'yolov8m.pt',
                 device: Optional[str] = None, fps_target: int = 30):
        """
        Initialize real-time inference.
        
        Args:
            checkpoint_path: Path to lightweight model checkpoint
            detection_model_path: Path to YOLO detection model
            device: Device to run on (None for auto-detect)
            fps_target: Target FPS for processing
        """
        self.runner = ModelRunner(
            model_type='lightweight',
            checkpoint_path=checkpoint_path,
            detection_model_path=detection_model_path,
            device=device
        )
        self.fps_target = fps_target
        self.frame_skip = max(1, 30 // fps_target)  # Skip frames if needed
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame and return distance prediction.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (distance_meters, metadata_dict) or (None, error_dict)
        """
        # Skip frames if needed to maintain target FPS
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return None, {'skipped': True}
        
        distance, metadata = self.runner.predict(frame)
        return distance, metadata
    
    def run_camera(self, camera_id: int = 0, callback: Optional[Callable] = None):
        """
        Run inference on live camera feed.
        
        Args:
            camera_id: Camera device ID
            callback: Optional callback function(distance, frame, metadata) called for each frame
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        print("Starting real-time inference. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                distance, metadata = self.process_frame(frame)
                
                # Draw results on frame
                if distance is not None:
                    cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Draw bounding box if available
                    if 'bbox' in metadata:
                        x1, y1, x2, y2 = map(int, metadata['bbox'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Call user callback if provided
                if callback:
                    callback(distance, frame, metadata)
                
                cv2.imshow('Distance Estimation', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_video_stream(self, video_path: str, output_path: Optional[str] = None,
                        callback: Optional[Callable] = None):
        """
        Run inference on a video file (for testing/debugging).
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video
            callback: Optional callback function(distance, frame, metadata) called for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        frame_num = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                distance, metadata = self.process_frame(frame)
                
                # Draw results
                if distance is not None:
                    cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if 'bbox' in metadata:
                        x1, y1, x2, y2 = map(int, metadata['bbox'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if callback:
                    callback(distance, frame, metadata)
                
                if writer:
                    writer.write(frame)
                
                frame_num += 1
                if frame_num % 30 == 0:
                    print(f"Processed {frame_num} frames...")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        print(f"Processed {frame_num} frames total.")



