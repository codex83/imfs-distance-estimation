"""
Heavyweight Model Architecture

3-branch hybrid model for distance estimation:
- Branch 1: Full image with EfficientNetB4 backbone
- Branch 2: Car patch with EfficientNetB3 backbone  
- Branch 3: Geometric features (10 features)

This is the main heavyweight model variant using EfficientNetB3 for the car patch branch.
"""

import torch
import torch.nn as nn
from torchvision import models


class HeavyweightModel(nn.Module):
    """
    Heavy 3-branch model (Heavyweight Model) for distance estimation.
    
    Architecture:
        1. Branch 1 (Full Image): EfficientNetB4 (pretrained, frozen initially)
        2. Branch 2 (Car Patch): EfficientNetB3 (pretrained, frozen initially)
        3. Branch 3 (Geometric): MLP processing 10 geometric features
    """
    
    def __init__(self, dropout_rate=0.3):
        super(HeavyweightModel, self).__init__()
        
        # Branch 1: Full Image - EfficientNetB4
        efficientnet_scene = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.full_image_backbone = nn.Sequential(*list(efficientnet_scene.children())[:-1])
        
        # Freeze EfficientNetB4 initially (can be unfrozen during training)
        for param in self.full_image_backbone.parameters():
            param.requires_grad = False
        
        self.full_image_head = nn.Sequential(
            nn.Linear(1792, 512),  # EfficientNetB4 outputs 1792 features
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Branch 2: Car Patch - EfficientNetB3
        efficientnet_patch = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.car_patch_backbone = nn.Sequential(*list(efficientnet_patch.children())[:-1])
        
        # Freeze EfficientNetB3 initially (can be unfrozen during training)
        for param in self.car_patch_backbone.parameters():
            param.requires_grad = False
        
        self.car_patch_head = nn.Sequential(
            nn.Linear(1536, 256),  # EfficientNetB3 outputs 1536 features
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Branch 3: Geometric features (10 features)
        self.geometric_branch = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion head
        # Input: 128 (full) + 64 (patch) + 32 (geo) = 224
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize fusion head weights."""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, full_image, car_patch, geometric):
        """
        Forward pass through the 3-branch model.
        
        Args:
            full_image: Full scene image tensor (batch, 3, H, W)
            car_patch: Cropped car patch tensor (batch, 3, H, W)
            geometric: Geometric features tensor (batch, 10)
            
        Returns:
            Distance prediction tensor (batch,)
        """
        # Process Branch 1: Full image
        full_features = self.full_image_backbone(full_image)
        full_features = full_features.view(full_features.size(0), -1)  # Flatten
        full_features = self.full_image_head(full_features)
        
        # Process Branch 2: Car patch
        patch_features = self.car_patch_backbone(car_patch)
        patch_features = patch_features.view(patch_features.size(0), -1)  # Flatten
        patch_features = self.car_patch_head(patch_features)
        
        # Process Branch 3: Geometric features
        geo_features = self.geometric_branch(geometric)
        
        # Combine all features
        combined = torch.cat([full_features, patch_features, geo_features], dim=1)
        
        # Final prediction
        output = self.fusion(combined)
        
        return output.squeeze(1)  # Squeeze from (batch, 1) to (batch,)
    
    def unfreeze_backbones(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.full_image_backbone.parameters():
            param.requires_grad = True
        for param in self.car_patch_backbone.parameters():
            param.requires_grad = True



