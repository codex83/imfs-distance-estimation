"""
Lightweight Model Architecture

3-branch lightweight model for distance estimation:
- Branch 1: Full image with MobileNetV3-Small backbone
- Branch 2: Car patch with Custom 4-Layer CNN
- Branch 3: Geometric features (10 features)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class LightweightModel(nn.Module):
    """
    Lightweight 3-branch model for edge deployment and real-time inference.
    
    Architecture:
        1. Branch 1 (Full Image): MobileNetV3-Small (pretrained)
        2. Branch 2 (Car Patch): Custom 4-Layer CNN
        3. Branch 3 (Geometric): Small MLP (shared with heavyweight model)
    """

    def __init__(self):
        super(LightweightModel, self).__init__()

        # --- Branch 1: Full Image (MobileNetV3-Small) ---
        # Load pretrained MobileNetV3-Small
        mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Extract features up to the avgpool layer, which outputs 576 features.
        self.full_image_backbone = nn.Sequential(
            mobilenet_v3.features,
            mobilenet_v3.avgpool
        )
        # Output shape from backbone: (batch, 576, 1, 1)

        # --- Branch 2: Car Patch (Custom 4-Layer CNN) ---
        self.car_patch_backbone = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Layer 4: 8x8 -> 8x8
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Global average pooling to flatten to (batch, 64)
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        # Output shape from backbone: (batch, 64, 1, 1)

        # --- Branch 3: Geometric Features (Shared with heavyweight model) ---
        self.geometric_branch = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Output shape from branch: (batch, 16)

        # --- Fusion Head ---
        # Input features: 576 (from B1) + 64 (from B2) + 16 (from B3) = 656
        self.fusion_head = nn.Sequential(
            nn.Linear(656, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output: 1 distance value
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
        # Process Branch 1
        full_features = self.full_image_backbone(full_image)
        full_features = torch.flatten(full_features, 1)  # Flatten to (batch, 576)

        # Process Branch 2
        patch_features = self.car_patch_backbone(car_patch)
        patch_features = torch.flatten(patch_features, 1)  # Flatten to (batch, 64)

        # Process Branch 3
        geo_features = self.geometric_branch(geometric)  # Output is (batch, 16)

        # Combine all features
        combined = torch.cat([full_features, patch_features, geo_features], dim=1)

        # Final prediction
        output = self.fusion_head(combined)

        return output.squeeze(1)  # Squeeze from (batch, 1) to (batch,)



