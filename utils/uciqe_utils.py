import numpy as np
import cv2

def getUCIQE(img):
    """
    Compute UCIQE (Underwater Color Image Quality Evaluation)
    img: RGB image in uint8 or float [0-255] or [0-1]
    Returns: normalized UCIQE value between 0 and 1
    """
    # Ensure uint8
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l, a, b = l.astype(np.float32), a.astype(np.float32), b.astype(np.float32)

    # Chroma
    chroma = np.sqrt((a - np.mean(a)) ** 2 + (b - np.mean(b)) ** 2)

    # σ_c: standard deviation of chroma
    sigma_c = np.std(chroma)

    # c_l: contrast of L (difference between 95th and 5th percentile)
    c_l = (np.percentile(l, 95) - np.percentile(l, 5)) / 255.0

    # μ_s: mean of saturation approximation
    s = np.sqrt(a ** 2 + b ** 2)
    mean_s = np.mean(s) / 255.0

    # UCIQE calculation
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe_val = c1 * sigma_c / 100.0 + c2 * c_l + c3 * mean_s

    # Normalize to 0–1 range
    uciqe_val = np.clip(uciqe_val, 0, 1)
    return round(float(uciqe_val), 4)
