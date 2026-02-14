Mamba-UIE Project
-----------------
This folder contains all code, dataset links, and configurations to train or evaluate the Mamba-UIE underwater image enhancement model on the UIEB dataset.

🧩 Setup Instructions
1. Install Python 3.9 or above.
2. Create a new environment and install dependencies:
   pip install -r requirements.txt

⚙️ Training (starts from scratch or resumes automatically)
   python main.py

📈 Evaluation (runs automatically after each epoch)
   - Saves up to 100 enhanced test images per epoch in subfolders.
   - Calculates PSNR, SSIM, UIQM, UCIQE, and MSE metrics.

📂 Folder Notes:
- Datasets/UIEB/
    ├── train_raw/          : Raw underwater training images
    ├── train_reference/    : Ground truth training images
    ├── test_raw/           : Raw underwater test images
    └── test_reference/     : Ground truth test images

- Training_Results_UIEB/
    ├── epoch_1.pth, epoch_2.pth, ...          : Model checkpoints per epoch
    ├── temp_epoch1_iter300.pth                : Mid-epoch backup checkpoints
    ├── epoch_1_samples/, epoch_2_samples/     : Saved comparison images (RAW | ENHANCED | REFERENCE)
    ├── metrics_log.csv                        : Epoch-wise metrics (Train/Val Loss, PSNR, SSIM, etc.)
    ├── loss_log.csv                           : Training and validation loss across epochs
    ├── *_vs_Epoch.png                         : Plots for all metrics vs. epoch

currently the weights of 51st epoch is saved so training will continue from 52nd. If you want to start training from beginning then delete that weight or change the location of save folder.

🧮 Metrics Tracked:
- PSNR, SSIM (Full-reference)
- UIQM, UCIQE (No-reference)
- MSE, Train/Validation Loss

🖼️ To Save Fewer Validation Images:
Inside `main.py`, search for:
   if i < min(100, len(test_loader)):
Change `100` to your desired number, e.g. `10` to save only 10 enhanced samples per epoch.

🧠 Environment Info:
- PyTorch 2.4.0 + CUDA 12.1
- NumPy, OpenCV, scikit-image, pandas, matplotlib
- Mamba-SSM 2.2.2
- causal-conv1d 1.4.0

📘 Notes:
- Training automatically resumes from the latest checkpoint found in the results folder.
- The results folder is created automatically at the start of training.
- All saved plots and logs are updated after each epoch.
