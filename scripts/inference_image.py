#!/usr/bin/env python3
"""
CLI script for single image inference.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference import OfflineInference


def main():
    parser = argparse.ArgumentParser(description='Run distance estimation on image(s)')
    
    # Model selection
    parser.add_argument('--model-type', type=str, choices=['heavyweight', 'lightweight', 'both'],
                       default='lightweight', help='Model type to use (default: lightweight)')
    
    # Checkpoints
    parser.add_argument('--checkpoint-path', type=str,
                       help='Path to model checkpoint (if using single model)')
    parser.add_argument('--heavyweight-checkpoint', type=str,
                       help='Path to heavyweight checkpoint (if using both models)')
    parser.add_argument('--lightweight-checkpoint', type=str,
                       help='Path to lightweight checkpoint (if using both models)')
    
    # Detection
    parser.add_argument('--detection-model', type=str, default='yolov8m.pt',
                       help='Path to YOLO detection model (default: yolov8m.pt)')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str,
                       help='Path to save annotated image or directory (optional)')
    parser.add_argument('--directory', action='store_true',
                       help='Process all images in input directory')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == 'both':
        if not args.heavyweight_checkpoint or not args.lightweight_checkpoint:
            parser.error("--heavyweight-checkpoint and --lightweight-checkpoint required for model-type=both")
    else:
        if not args.checkpoint_path:
            parser.error("--checkpoint-path required for single model type")
    
    # Initialize inference
    inference = OfflineInference(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        heavyweight_checkpoint_path=args.heavyweight_checkpoint,
        lightweight_checkpoint_path=args.lightweight_checkpoint,
        detection_model_path=args.detection_model,
        device=args.device
    )
    
    # Process image(s)
    if args.directory:
        results = inference.process_directory(
            input_dir=args.input,
            output_dir=args.output
        )
        print(f"\nProcessed {len(results)} images")
    else:
        result = inference.process_image(
            image_path=args.input,
            output_path=args.output
        )
        print(f"\nResult: {result}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())



