import torch.nn as nn


class YourCustomModelClassName(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, latent_dim=128):
        """
        CSI (Channel State Information) classification model that fuses spatial feature extraction and temporal dependency capture for classification tasks.

        Args:
            num_classes (int, optional): Number of target classes for the classification task, default is 10.
            base_channels (int, optional): Base number of channels for convolutional layers, which controls the feature dimension scale of the spatial encoder, default is 32.
            latent_dim (int, optional): Latent feature dimension of the temporal processor, also serving as the d_model of TransformerEncoderLayer, default is 128.
        """
        super().__init__()

        self.num_classes = num_classes
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, groups=base_channels),
            nn.Conv2d(base_channels * 2, base_channels * 2, 1),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.temporal_processor = nn.Sequential(
            nn.Linear(base_channels * 2 * 16, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim * 2,
                dropout=0.2,
                batch_first=True
            )
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.output_layer = nn.Linear(latent_dim, self.num_classes)

    def forward(self, x):
        """
        Forward propagation logic of the CSI classification model.

        Args:
            x (torch.Tensor): Input tensor with shape (B, T, F, A), where B = batch size, T = time steps, F = number of CSI subcarriers, A = number of antennas.

        Returns:
            torch.Tensor: Output tensor with shape (B, num_classes), representing the prediction scores for each class of each sample.
        """
        B, T, F, A = x.shape

        spatial_input = x.view(B * T, 1, F, A)  # (B, T, F, A) -> (B*T, 1, F, A)
        spatial_feat = self.spatial_encoder(spatial_input)  # (B*T, 1, F, A) -> (B*T, base_channels*2 * 16)
        spatial_feat = spatial_feat.view(B, T, -1)  # (B*T, base_channels*2 * 16) -> (B, T, base_channels*2 * 16)

        temporal_feat = self.temporal_processor(spatial_feat)  # (B, T, base_channels*2 * 16) -> (B, T, latent_dim)
        pooled = self.adaptive_pool(
            temporal_feat.transpose(1, 2))  # (B, T, latent_dim) -> (B, latent_dim, T) -> (B, latent_dim, 1)
        features = self.flatten(pooled)  # (B, latent_dim, 1) -> (B, latent_dim)

        return self.output_layer(features)  # (B, latent_dim) -> (B, num_classes)

model = YourCustomModelClassName
