"""
Lightweight Model Training

Training script for the lightweight model with MobileNetV3-Small + Custom CNN backbones.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from .common import train_one_epoch, validate
from ..models import LightweightModel
from ..data import create_geometric_dataloaders


def train_lightweight_model(
    detection_json_dir,
    train_image_dir,
    val_image_dir,
    test_image_dir,
    checkpoint_dir,
    num_epochs=50,
    learning_rate=0.001,
    batch_size=64,
    num_workers=8,
    device=None,
    use_torch_compile=False
):
    """
    Train the lightweight model.
    
    Args:
        detection_json_dir: Directory containing train/val/test detection JSON files
        train_image_dir: Directory containing training images
        val_image_dir: Directory containing validation images
        test_image_dir: Directory containing test images
        checkpoint_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        device: Device to train on (None for auto-detect)
        use_torch_compile: Whether to use torch.compile for optimization
        
    Returns:
        Training history dictionary
    """
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = LightweightModel().to(device)
    
    # Optional: Compile model for optimization (PyTorch 2.0+)
    if use_torch_compile and device.type == 'cuda':
        model = torch.compile(model)
        print("Model compiled for optimization")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Create dataloaders
    print(f"\nLoading data...")
    print(f"  JSONs from: {detection_json_dir}")
    print(f"  Train images from: {train_image_dir}")
    print(f"  Val images from: {val_image_dir}")
    print(f"  Test images from: {test_image_dir}")
    
    train_loader, val_loader, test_loader = create_geometric_dataloaders(
        detection_json_dir=detection_json_dir,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Mixed precision training
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.amp.GradScaler('cpu')
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING LIGHTWEIGHT MODEL TRAINING")
    print("="*70)
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'lr': []
    }
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Log history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.4f}m | Val RMSE: {val_rmse:.4f}m")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model_lightweight.pt')
            print(f"  ✨ New best model found! Saving to {checkpoint_path}")
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse
            }
            torch.save(state, checkpoint_path)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training Complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history



