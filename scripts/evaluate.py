#!/usr/bin/env python3
"""
CLI script for evaluating models.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import compare_models, evaluate_model
from src.models import HeavyweightModel, LightweightModel
from src.data import create_geometric_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Evaluate distance estimation models')
    
    parser.add_argument('--mode', type=str, choices=['single', 'compare'], default='compare',
                       help='Evaluation mode: single model or compare both (default: compare)')
    
    # Model selection
    parser.add_argument('--model-type', type=str, choices=['heavyweight', 'lightweight'],
                       help='Model type for single mode (required if mode=single)')
    parser.add_argument('--checkpoint-path', type=str,
                       help='Path to model checkpoint (required if mode=single)')
    
    # Checkpoints for comparison
    parser.add_argument('--heavyweight-checkpoint', type=str,
                       help='Path to heavyweight checkpoint (required if mode=compare)')
    parser.add_argument('--lightweight-checkpoint', type=str,
                       help='Path to lightweight checkpoint (required if mode=compare)')
    
    # Data paths
    parser.add_argument('--detection-json-dir', type=str, required=True,
                       help='Directory containing test detection JSON file')
    parser.add_argument('--test-image-dir', type=str, required=True,
                       help='Directory containing test images')
    
    # Dataloader config
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers (default: 8)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create test dataloader
    _, _, test_loader = create_geometric_dataloaders(
        detection_json_dir=args.detection_json_dir,
        train_image_dir=args.test_image_dir,  # Dummy, not used
        val_image_dir=args.test_image_dir,    # Dummy, not used
        test_image_dir=args.test_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if args.mode == 'compare':
        if not args.heavyweight_checkpoint or not args.lightweight_checkpoint:
            parser.error("--heavyweight-checkpoint and --lightweight-checkpoint required for mode=compare")
        
        heavyweight_model = HeavyweightModel()
        lightweight_model = LightweightModel()
        
        results = compare_models(
            heavyweight_model=heavyweight_model,
            lightweight_model=lightweight_model,
            test_loader=test_loader,
            heavyweight_checkpoint_path=args.heavyweight_checkpoint,
            lightweight_checkpoint_path=args.lightweight_checkpoint,
            device=args.device
        )
    else:
        if not args.model_type or not args.checkpoint_path:
            parser.error("--model-type and --checkpoint-path required for mode=single")
        
        if args.model_type == 'heavyweight':
            model = HeavyweightModel()
        else:
            model = LightweightModel()
        
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=args.device,
            model_name=f"{args.model_type.capitalize()} Model"
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())



