# Improved Mamba-UIE: Learnable GBL and Adaptive Loss for Underwater Image Enhancement

This project enhances a physics-aware underwater image restoration framework by introducing learnable illumination modeling and adaptive optimization strategies.

The objective of this work is to improve physical consistency, optimization stability, and perceptual quality without increasing architectural complexity.

---

## 🚀 Key Improvements

### 1️⃣ Learnable Global Background Light (GBL)

Replaced heuristic background light estimation with a trainable module.

- Global average pooling for scene-level color statistics  
- Lightweight MLP for illumination prediction  
- Sigmoid-bounded output to preserve physical validity  
- Regularization for stable optimization  

Updated formation model:

I'(x) = J(x)TD(x) + (1 − TB(x))Aθ

---

### 2️⃣ Adaptive Softmax-Based Loss Reweighting

Introduced dynamic loss balancing instead of fixed manual weights:

w_i = exp(-αL_i) / Σ exp(-αL_j)

Benefits:
- Automatic balancing of multi-term losses  
- Reduced hyperparameter tuning  
- Curriculum-style training behavior  
- Improved convergence stability  

Cosine annealing temperature scheduling applied for smooth optimization.

---

### 3️⃣ Smooth L1 Reconstruction Loss

Replaced standard L2 reconstruction loss with Smooth L1 to:

- Reduce sensitivity to outliers  
- Stabilize transmission map prediction  
- Prevent gradient explosion  
- Improve structural preservation  

---

### 4️⃣ Curved Channel Attention (CCA)

Introduced wavelength-aware channel attention to model nonlinear RGB attenuation in underwater environments, improving illumination consistency and color restoration.

---

## 📊 Quantitative Results (UIEB Dataset)

| Metric | Before | After |
|--------|--------|-------|
| PSNR   | 23.50  | 23.93 |
| SSIM   | 0.9049 | 0.9198 |
| UIQM   | 3.148  | 3.242 |

### Performance Improvements:
- +0.43 dB PSNR  
- +0.015 SSIM  
- +0.093 UIQM  

All improvements achieved without increasing model complexity.

---

## ⚙️ Training Setup

- Dataset: UIEB (800 training / 90 validation images)  
- Input Resolution: 256 × 256  
- Optimizer: Adam (1e-4)  
- Scheduler: Cosine Annealing  
- Epochs: 50  
- Batch Size: 1  
- GPU: NVIDIA T4  

---

The complete technical documentation of this work is available in the repository:

📄 **DL_REPORT_MAMBA_UIE_EE23BT021.pdf**

The report includes:

- Mathematical formulation of the updated image formation model  
- Implementation details of the learnable GBL module  
- Adaptive loss reweighting derivation  
- Training configuration and hyperparameter selection  
- Quantitative evaluation and ablation studies  


