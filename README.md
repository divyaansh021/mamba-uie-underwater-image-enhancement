# Improved Mamba-UIE: Learnable GBL and Adaptive Loss for Underwater Image Enhancement

This project enhances a physics-aware underwater image restoration framework by introducing learnable illumination modeling and adaptive optimization strategies.

The focus of this work is improving physical consistency, training stability, and perceptual quality without increasing model complexity.

---

## 🚀 Key Improvements

### 1️⃣ Learnable Global Background Light (GBL)

Replaced heuristic background light estimation with a trainable module.

- Uses global average pooling for scene-level color statistics
- Lightweight MLP for illumination prediction
- Sigmoid-bounded output for physical validity
- Regularization for stable training

Updated formation model:

I'(x) = J(x)TD(x) + (1 − TB(x))Aθ

---

### 2️⃣ Adaptive Softmax-Based Loss Reweighting

Introduced dynamic loss balancing instead of fixed manual weights.

w_i = exp(-αL_i) / Σ exp(-αL_j)

Benefits:
- Automatic balancing of loss components
- Reduced hyperparameter tuning
- Curriculum-style training behavior
- Improved optimization stability

Cosine annealing temperature scheduling applied for smooth convergence.

---

### 3️⃣ Smooth L1 Reconstruction Loss

Replaced standard L2 reconstruction loss with Smooth L1 to:

- Reduce sensitivity to outliers
- Stabilize transmission map prediction
- Prevent gradient explosion
- Improve edge preservation

---

### 4️⃣ Curved Channel Attention (CCA)

Introduced wavelength-aware channel attention to model nonlinear RGB attenuation in underwater environments, improving color consistency.

---

## 📊 Quantitative Results (UIEB Dataset)

| Metric | Before | After |
|--------|--------|-------|
| PSNR   | 23.50  | 23.93 |
| SSIM   | 0.9049 | 0.9198 |
| UIQM   | 3.148  | 3.242 |

### Improvements:
- +0.43 dB PSNR
- +0.015 SSIM
- +0.093 UIQM

Performance gains achieved without increasing architectural complexity.

---

## ⚙️ Training Setup

- Dataset: UIEB (800 train / 90 validation)
- Input Resolution: 256×256
- Optimizer: Adam (1e-4)
- Scheduler: Cosine Annealing
- Epochs: 50
- Batch Size: 1
- GPU: NVIDIA T4

---



