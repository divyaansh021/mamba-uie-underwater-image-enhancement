

# ============================================================
# Added CNN helper blocks for Mamba-UIE (ResBlock, ConvBlock, etc.)
# ============================================================

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x):
        return x + self.layer(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x):
        return self.up(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
    def forward(self, x):
        return self.shuffle(self.conv(x))

class Compute_z(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=True)
