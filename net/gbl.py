import torch
import torch.nn as nn
import torch.nn.functional as F

class CurvedGBLNet(nn.Module):
    """
    Learnable Global Background Light Estimation
    using channel-wise curved attenuation + spatial split + fusion convs.
    """
    def __init__(self):
        super(CurvedGBLNet, self).__init__()

        # Each color channel learns nonlinear attenuation separately
        self.branch_r = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.branch_g = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Fusion (1×1 conv → global RGB background light)
        self.fuse = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Split into R, G, B channels
        r, g, b = torch.chunk(x, 3, dim=1)

        # Channel-wise nonlinear attenuation
        r_att = self.branch_r(r)
        g_att = self.branch_g(g)
        b_att = self.branch_b(b)

        # Combine and fuse
        A_map = torch.cat([r_att, g_att, b_att], dim=1)
        A = self.fuse(A_map)  # [B, 3, 1, 1]
        return A
