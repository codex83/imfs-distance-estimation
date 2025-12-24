#!/usr/bin/env python3
"""
CLI script for training the heavyweight model.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import train_heavyweight_model


def main():
    parser = argparse.ArgumentParser(description='Train heavyweight model')
    
    # Data paths
    parser.add_argument('--detection-json-dir', type=str, required=True,
                       help='Directory containing train/val/test detection JSON files')
    parser.add_argument('--train-image-dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--val-image-dir', type=str, required=True,
                       help='Directory containing validation images')
    parser.add_argument('--test-image-dir', type=str, required=True,
                       help='Directory containing test images')
    
    # Training config
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--unfreeze-epoch', type=int, default=10,
                       help='Epoch to unfreeze backbones (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers (default: 8)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Train model
    history = train_heavyweight_model(
        detection_json_dir=args.detection_json_dir,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        test_image_dir=args.test_image_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        unfreeze_epoch=args.unfreeze_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )
    
    print("\nTraining completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())



