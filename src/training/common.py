"""
Common Training Utilities

Shared training functions used by both heavyweight and lightweight model training.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision training
        device: Device to run on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    loop = tqdm(loader, total=len(loader), leave=True)
    for inputs, targets in loop:
        full_imgs = inputs['full_image'].to(device, non_blocking=True)
        car_patches = inputs['car_patch'].to(device, non_blocking=True)
        geo_feats = inputs['geometric'].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(device_type=device.type):
            outputs = model(full_imgs, car_patches, geo_feats)
            loss = criterion(outputs, targets)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Validate the model on a dataset.
    
    Args:
        model: The model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (validation_loss, MAE, RMSE)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            full_imgs = inputs['full_image'].to(device, non_blocking=True)
            car_patches = inputs['car_patch'].to(device, non_blocking=True)
            geo_feats = inputs['geometric'].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Mixed precision inference
            with autocast(device_type=device.type):
                outputs = model(full_imgs, car_patches, geo_feats)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(loader)
    val_mae = mean_absolute_error(all_targets, all_preds)
    val_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    return val_loss, val_mae, val_rmse



