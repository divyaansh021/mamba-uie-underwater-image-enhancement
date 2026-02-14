import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x):  ## 灰度世界
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.pow(mr-0.5, 2)
        Dg = torch.pow(mg-0.5, 2)
        Db = torch.pow(mb-0.5, 2)
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)
        return k

class ColorLossImproved(nn.Module):
    def __init__(self):
        super(ColorLossImproved, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.abs(mr-0.5)
        Dg = torch.abs(mg-0.5)
        Db = torch.abs(mb-0.5)
        k = torch.pow(Dr+Dg+Db, 2)
        return k

def histogram_spread(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 1))
    return np.std(hist)

class ColorLoss1(nn.Module):
    def __init__(self):
        super(ColorLoss1, self).__init__()

    def forward(self, x):

        ## 数据预处理
        x_np = x.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        # Convert from RGB to BGR if needed
        input_img = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

        ## zip [(img_mean, img)], it (b, g, r)
        small, medium, large = sorted(list(zip(cv2.mean(input_img), cv2.split(input_img), ['b', 'g', 'r'])))
        ## sorted by mean (small to large)
        small, medium, large = list(small), list(medium), list(large)

        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':  ### 同时满足三个条件
            large, medium = medium, large  ## 中等 和大 交换

        loss = np.sqrt((large[0] - cv2.mean(medium[1])[0])**2 + (large[0] - cv2.mean(small[1])[0])**2)

        return loss


def RecoverCLAHE(sceneRadiance):
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8)) ## re-waternet中的设置
    # clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4, 4))
    for i in range(3):

        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))

    return sceneRadiance

def tensor_to_cv2_img(tensor_img):
    # 将 PyTorch 张量的形状转换为 (h, w, 3)
    # img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 转换数据类型为 uint8
    img_np = (img_np * 255).astype('uint8')
    return img_np


def cv2_img_to_tensor(cv2_img):
    # 将 cv2 格式的图像数据转换为 PyTorch 张量
    tensor_img = torch.tensor(cv2_img, dtype=torch.float32)  # 将数据类型转换为 float32
    # 将通道顺序从 BGR 转换为 RGB
    tensor_img = tensor_img.permute(2, 0, 1)

    # 将数据范围从 [0, 255] 转换为 [0, 1]
    # tensor_img /= 255.0
    # 添加批次维度
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img

def CLAHE_loss(img):  ## 损失不下降？？

    img_cv2 = tensor_to_cv2_img(img)
    CLAHE = RecoverCLAHE(img_cv2)
    CLAHE_tensor = cv2_img_to_tensor(CLAHE)

    mse_loss = nn.MSELoss()
    clahe_loss = mse_loss(img, CLAHE_tensor)
    return clahe_loss

def contrast_loss(image):
    # 计算图像梯度
    gradient_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    gradient_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    # 对梯度进行平滑处理，以减少噪音
    gradient_x_smooth = F.avg_pool2d(gradient_x, kernel_size=3, stride=1, padding=(0, 1))
    gradient_y_smooth = F.avg_pool2d(gradient_y, kernel_size=3, stride=1, padding=(1, 0))

    # 计算梯度的均值，作为对比度损失
    contrast_loss = torch.mean(gradient_x_smooth) + torch.mean(gradient_y_smooth)

    return contrast_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss# ==========================================================
# ✅ Adaptive Weighted Smooth L1 Loss for Mamba-UIE
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, alpha=5.0, beta=1.0, lambda_edge=0.05):
        """
        Adaptive weighted total loss for Mamba-UIE.
        - Replaces L2 with Smooth L1.
        - Keeps λ (lambda_edge) fixed for edge loss.
        - Dynamically adjusts weights for all other loss components.
        """
        super(AdaptiveWeightedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_edge = lambda_edge

    def forward(self, losses):
        """
        losses = [l_smooth, l_ssim, l_edge, l_uiqm, l_smooth_r, l_ssim_r]
        """
        l_smooth, l_ssim, l_edge, l_uiqm, l_smooth_r, l_ssim_r = losses

        # Create adaptive weights (exclude edge)
        l_vector = torch.tensor(
            [l_smooth, l_ssim, l_uiqm, l_smooth_r, l_ssim_r],
            device=l_smooth.device
        )
        weights = F.softmax(-self.alpha * l_vector.detach(), dim=0)
        w1, w2, w3, w4, w5 = weights

        # Combine total loss (λ fixed)
        total_loss = (
            w1 * l_smooth +
            w2 * l_ssim +
            self.lambda_edge * l_edge +
            w3 * l_uiqm +
            w4 * l_smooth_r +
            w5 * l_ssim_r
        )

        return total_loss, weights


"""
📘 Example Usage (inside train() in main.py):

Edge = EdgeLoss()
adaptive_loss = AdaptiveWeightedLoss(alpha=5.0, lambda_edge=0.05).cuda()

l_smooth = F.smooth_l1_loss(j_out, label, beta=1.0)
l_ssim = 1 - ssim(j_out, label)
l_edge = Edge(label, j_out)
l_uiqm = 1 / getUIQM((j_out[0].detach().cpu().permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8))
l_smooth_r = F.smooth_l1_loss(I_rec, input, beta=1.0)
l_ssim_r = 1 - ssim(I_rec, input)

loss, adaptive_weights = adaptive_loss([l_smooth, l_ssim, l_edge, l_uiqm, l_smooth_r, l_ssim_r])
print(f"Adaptive Weights: {[round(w.item(), 4) for w in adaptive_weights]}")
"""
