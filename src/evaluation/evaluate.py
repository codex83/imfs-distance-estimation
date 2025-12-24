"""
Model Evaluation Utilities

Functions for evaluating distance estimation models on test datasets.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import autocast


def run_inference(model, loader, device):
    """
    Run inference on a model and collect predictions and targets.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for the dataset
        device: Device to run on
        
    Returns:
        Tuple of (mae, rmse, r2, predictions, targets)
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Running Inference"):
            full_imgs = inputs['full_image'].to(device, non_blocking=True)
            car_patches = inputs['car_patch'].to(device, non_blocking=True)
            geo_feats = inputs['geometric'].to(device, non_blocking=True)
            
            # Mixed precision inference
            with autocast(device_type=device.type):
                outputs = model(full_imgs, car_patches, geo_feats)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return mae, rmse, r2, all_preds, all_targets


def evaluate_model(model, test_loader, device=None, model_name="Model"):
    """
    Evaluate a model on a test dataset and print metrics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to run on (None for auto-detect)
        model_name: Name of the model for printing
        
    Returns:
        Dictionary with metrics and predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    print(f"Using device: {device}")
    
    mae, rmse, r2, predictions, targets = run_inference(model, test_loader, device)
    
    print(f"\nResults:")
    print(f"  MAE:  {mae:.4f}m")
    print(f"  RMSE: {rmse:.4f}m")
    print(f"  R²:   {r2:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def compare_models(heavyweight_model, lightweight_model, test_loader, 
                   heavyweight_checkpoint_path=None, lightweight_checkpoint_path=None,
                   device=None):
    """
    Compare heavyweight and lightweight models on the test set.
    
    Args:
        heavyweight_model: Heavyweight model instance
        lightweight_model: Lightweight model instance
        test_loader: DataLoader for test data
        heavyweight_checkpoint_path: Path to heavyweight checkpoint (optional)
        lightweight_checkpoint_path: Path to lightweight checkpoint (optional)
        device: Device to run on (None for auto-detect)
        
    Returns:
        Dictionary with comparison results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load checkpoints if provided
    if heavyweight_checkpoint_path:
        print(f"Loading heavyweight checkpoint from {heavyweight_checkpoint_path}")
        checkpoint = torch.load(heavyweight_checkpoint_path, map_location=device)
        heavyweight_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    if lightweight_checkpoint_path:
        print(f"Loading lightweight checkpoint from {lightweight_checkpoint_path}")
        checkpoint = torch.load(lightweight_checkpoint_path, map_location=device)
        lightweight_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Move models to device
    heavyweight_model = heavyweight_model.to(device)
    lightweight_model = lightweight_model.to(device)
    
    # Evaluate both models
    heavyweight_results = evaluate_model(heavyweight_model, test_loader, device, "Heavyweight Model")
    lightweight_results = evaluate_model(lightweight_model, test_loader, device, "Lightweight Model")
    
    # Print comparison
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON (on Test Set)")
    print("="*70)
    print(f"| Model                 | Test MAE (meters) | Test RMSE (meters) | R² Score |")
    print(f"|-----------------------|-------------------|--------------------|----------|")
    print(f"| Heavyweight           | {heavyweight_results['mae']:^17.4f} | {heavyweight_results['rmse']:^18.4f} | {heavyweight_results['r2']:^8.4f} |")
    print(f"| Lightweight           | {lightweight_results['mae']:^17.4f} | {lightweight_results['rmse']:^18.4f} | {lightweight_results['r2']:^8.4f} |")
    print("="*70)
    
    mae_diff = lightweight_results['mae'] - heavyweight_results['mae']
    rmse_diff = lightweight_results['rmse'] - heavyweight_results['rmse']
    r2_diff = lightweight_results['r2'] - heavyweight_results['r2']
    
    print(f"\nDifference (Lightweight vs Heavyweight):")
    print(f"  MAE:  {mae_diff:+.4f}m")
    print(f"  RMSE: {rmse_diff:+.4f}m")
    print(f"  R²:   {r2_diff:+.4f}")
    print("="*70 + "\n")
    
    return {
        'heavyweight': heavyweight_results,
        'lightweight': lightweight_results,
        'differences': {
            'mae': mae_diff,
            'rmse': rmse_diff,
            'r2': r2_diff
        }
    }



