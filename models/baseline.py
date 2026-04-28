import torch
import torch.nn as nn

from src import config


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        spatial_channels = config.PATCH_MODEL_SPATIAL_CHANNELS
        temporal_channels = config.PATCH_MODEL_TEMPORAL_CHANNELS

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, spatial_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_channels // 2),
            nn.ELU(inplace=True),
            nn.Conv2d(spatial_channels // 2, spatial_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.temporal_stem = nn.Sequential(
            nn.Conv1d(spatial_channels, temporal_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.ELU(inplace=True),
        )
        self.temporal_block1 = TemporalBlock(temporal_channels, temporal_channels)
        self.temporal_block2 = TemporalBlock(temporal_channels, temporal_channels)

        self.roi_attention = nn.Linear(temporal_channels, 1)
        self.head = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(temporal_channels // 2),
            nn.ELU(inplace=True),
            nn.Conv1d(temporal_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(f"Model expects input shaped [batch, time, roi, 3, h, w], got {tuple(x.shape)}")

        batch_size, time_steps, roi_count, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel patches, got {channels}")

        spatial = x.reshape(batch_size * time_steps * roi_count, channels, height, width)
        spatial = self.spatial_encoder(spatial).flatten(1)
        spatial = spatial.view(batch_size, time_steps, roi_count, -1)

        temporal = spatial.permute(0, 2, 3, 1).reshape(batch_size * roi_count, spatial.shape[-1], time_steps)
        temporal = self.temporal_stem(temporal)
        temporal = self.temporal_block1(temporal)
        temporal = self.temporal_block2(temporal)
        temporal = temporal.view(batch_size, roi_count, temporal.shape[1], time_steps)

        roi_summary = temporal.mean(dim=-1)
        roi_weights = torch.softmax(self.roi_attention(roi_summary).squeeze(-1), dim=1)
        fused = (temporal * roi_weights[:, :, None, None]).sum(dim=1)

        out = self.head(fused)
        return out[:, 0, :]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))
