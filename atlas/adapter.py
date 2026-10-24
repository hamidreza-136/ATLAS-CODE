"""Attention adaptation network (Sec. 3.3, Table 10).

A two-stage U-Net maps the stacked attention of each keyframe to a
localization logit map. At the bottleneck, global-average-pooled features of
all keyframes go through a temporal Transformer whose outputs are projected to
FiLM parameters that modulate each keyframe's bottleneck features.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    """Transposed-convolution upsampling, concatenation with the skip, conv block."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        self.conv = conv_block(2 * c_out, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align to the skip resolution (odd grid sizes, and the bottleneck stage
        # whose skip shares the bottleneck resolution, Table 10).
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionAdapter(nn.Module):
    """U-Net with temporal FiLM conditioning.

    Input:  ``(B, T, 2C, H, W)`` frame + video attention of ``T`` keyframes,
            where clip ``i`` only uses its first ``lengths[i]`` keyframes.
    Output: ``(sum(lengths), H, W)`` localization logits of the valid keyframes,
            clip by clip (``P_k = sigmoid(logits)``).
    """

    def __init__(self, in_channels: int, base: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        d = 4 * base  # bottleneck dimension (512)
        self.config = dict(in_channels=in_channels, base=base, num_layers=num_layers, num_heads=num_heads)

        self.enc1 = conv_block(in_channels, base)        # H x W
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, 2 * base)           # H/2 x W/2
        self.bottleneck = conv_block(2 * base, d)        # H/2 x W/2

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=4 * d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.film = nn.Linear(d, 2 * d)                  # [W_gamma; W_beta], Eq. (7)

        self.up2 = Up(d, 2 * base)                       # fuse with enc2
        self.up1 = Up(2 * base, base)                    # fuse with enc1
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor, lengths: Optional[List[int]] = None) -> torch.Tensor:
        b, t, c, h, w = x.shape
        if lengths is None:
            lengths = [t] * b
        valid = torch.arange(t, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        x = x.reshape(b * t, c, h, w)[valid.flatten()]   # padded keyframes are dropped

        # Encoder (Eq. 4).
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        z = self.bottleneck(s2)                          # Z_k: (N, d, h/2, w/2)

        # Temporal context over the keyframes of each clip (Eq. 5-6).
        tokens = z.new_zeros(b, t, z.shape[1])
        tokens[valid] = z.mean(dim=(-2, -1))             # z_k (GAP)
        context = self.temporal(tokens, src_key_padding_mask=~valid)[valid]

        # FiLM modulation of the bottleneck (Eq. 7-8).
        gamma, beta = self.film(context)[:, :, None, None].chunk(2, dim=1)
        z = z * (1 + gamma) + beta

        # Decoder (Eq. 9).
        y = self.up1(self.up2(z, s2), s1)
        return self.head(y)[:, 0]


def adapter_input(frame_attn: torch.Tensor, video_attn: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    """Eq. (3): resize both branches to ``grid`` and concatenate them channel-wise.

    ``frame_attn`` / ``video_attn``: ``(T, C, h, w)`` attention of one clip
    (the video branch has one temporal slot per keyframe). The frame grid
    depends on the frame aspect ratio, so every clip is brought to the grid
    the adapter is trained on. Returns ``(T, 2C, *grid)``.
    """
    def resize(a):
        return F.interpolate(a.float(), size=grid, mode="bilinear", align_corners=False)
    return torch.cat([resize(frame_attn), resize(video_attn)], dim=1)


def load_adapter(path: str, device: str = "cuda") -> Tuple[AttentionAdapter, Tuple[int, int]]:
    """Load a trained adapter and the attention grid ``(H, W)`` it was trained on."""
    ckpt = torch.load(path, map_location="cpu")
    model = AttentionAdapter(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), tuple(ckpt["grid"])
