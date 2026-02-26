import torch
from torch import nn
import torch.nn.functional as F

class MultimodalPICAINet(nn.Module):

    def __init__(self, metadata_input_dim):

        super().__init__()

        self.custom_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.LeakyReLU(1e-1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2), # 32x32

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(1e-1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2), # 16x16

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(1e-1),
            nn.BatchNorm2d(16)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.image_feature_dim = 16 * 1 * 1
        
        self.attn_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(1e-1),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, 8),
            nn.LeakyReLU(1e-1),
            nn.Linear(8, 16),
            nn.LeakyReLU(1e-1),
            nn.Linear(16, 8),
            nn.LeakyReLU(1e-1),
            nn.Dropout(0.2)
        )
        self.metadata_embedding_dim = 8

        self.multimodal_dim = self.image_feature_dim + self.metadata_embedding_dim

        self.multimodal_classifier = nn.Sequential(
            nn.Linear(self.multimodal_dim, 16),
            nn.LeakyReLU(1e-1),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        
        self.single_modal_classifier = nn.Sequential(
            nn.Linear(self.image_feature_dim, 16),
            nn.LeakyReLU(1e-1),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        
    def forward(self, image, lesion_mask, metadata):

        # --- Custom Backbone Stem ---

        x = self.custom_backbone(image)

        # # Resize mask to match feature map
        # mask_resized = F.interpolate(
        #     lesion_mask,
        #     size=x.shape[-2:],
        #     mode='nearest'
        # )

        # # Learnable attention
        # attention = self.attn_conv(mask_resized)

        # # Apply attention
        # x = x * (1 + attention)

        x = self.global_avg_pool(x)

        image_features = torch.flatten(x, 1)

        # # Metadata branch
        # metadata_embeddings = self.metadata_mlp(metadata)

        # # Multimodal fusion
        # combined_features = torch.cat([image_features, metadata_embeddings], dim=1)
        # output = self.multimodal_classifier(combined_features).squeeze(1)

        output = self.single_modal_classifier(image_features).squeeze(1)
        
        return output