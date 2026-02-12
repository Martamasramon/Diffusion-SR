import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class MultimodalPICAINet(nn.Module):

    def __init__(self, metadata_input_dim):

        super().__init__()

        backbone = models.resnet50(pretrained=True)

        # for param in backbone.parameters():
        #     param.requires_grad = False

        for name, param in backbone.named_parameters():
            if not ("layer3" in name or "layer4" in name):
                param.requires_grad = False

        in_features_fc = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.image_feature_dim = in_features_fc

        self.attn_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.metadata_embedding_dim = 32

        self.multimodal_dim = self.image_feature_dim + self.metadata_embedding_dim
        self.multimodal_classifier = nn.Sequential(
            nn.Linear(self.multimodal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
        
    def forward(self, image, lesion_mask, metadata):

        # --- ResNet Stem ---
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # --- Layer 1 ---
        x = self.backbone.layer1(x)

        # Resize mask to match feature map
        mask_resized = F.interpolate(
            lesion_mask,
            size=x.shape[-2:],
            mode='nearest'
        )

        # Learnable attention
        attention = self.attn_conv(mask_resized)

        # Apply attention
        x = x * (1 + attention)

        # Continue backbone
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        image_features = torch.flatten(x, 1)

        # Metadata branch
        metadata_embeddings = self.metadata_mlp(metadata)

        # Multimodal fusion
        combined_features = torch.cat([image_features, metadata_embeddings], dim=1)
        output = self.multimodal_classifier(combined_features).squeeze(1)

        return output