"""
Heavyweight Model V2 (MobileNetV2 Variant)

This is the original heavyweight model variant using MobileNetV2 for the car patch branch.
Kept for backward compatibility and comparison purposes.
"""

import torch
import torch.nn as nn
from torchvision import models


class HeavyweightModelV2(nn.Module):
    """
    Original Heavyweight Model variant: EfficientNetB4 + MobileNetV2 + Geometric Features
    
    This version uses MobileNetV2 for the car patch branch (lighter than EfficientNetB3).
    """
    
    def __init__(self):
        super(HeavyweightModelV2, self).__init__()
        
        # Branch 1: Full image with EfficientNetB4
        efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        efficientnet.classifier = nn.Identity()
        self.full_image_backbone = efficientnet
        
        # Freeze EfficientNetB4 initially (can be unfrozen during training)
        for param in self.full_image_backbone.parameters():
            param.requires_grad = False
        
        self.full_image_head = nn.Sequential(
            nn.Linear(1792, 128),  # EfficientNetB4 outputs 1792 features
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branch 2: Car patch with MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        mobilenet.classifier = nn.Identity()
        self.car_patch_backbone = mobilenet
        
        # Freeze MobileNetV2 initially (can be unfrozen during training)
        for param in self.car_patch_backbone.parameters():
            param.requires_grad = False
        
        self.car_patch_head = nn.Sequential(
            nn.Linear(1280, 128),  # MobileNetV2 outputs 1280 features
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branch 3: Geometric features (10 features)
        self.geometric_branch = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 16, 128),  # 272 features total
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
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
        full_features = self.full_image_head(full_features)
        
        # Process Branch 2: Car patch
        patch_features = self.car_patch_backbone(car_patch)
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

