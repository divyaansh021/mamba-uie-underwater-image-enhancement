import os
import random
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils.SSIM_loss import ssim
from utils.uqim_utils import getUIQM
from utils.uciqe_utils import getUCIQE
from data import get_training_set, get_eval_set
from net.net import net
# from A import get_A  # ❌ removed static background light
from net.losses import *
import argparse
import gc

# ======================
# Training settings
# ======================
parser = argparse.ArgumentParser(description='PyTorch Mamba-UIE (EUVP Training + Evaluation)')
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--nEpochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--decay', type=int, default=200)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--rgb_range', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=256)



# ======================
# Portable local paths for UIEB
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser.add_argument('--data_train', type=str, default=os.path.join(BASE_DIR, 'Datasets', 'UIEB', 'train_raw'))
parser.add_argument('--label_train', type=str, default=os.path.join(BASE_DIR, 'Datasets', 'UIEB', 'train_reference'))
parser.add_argument('--data_test',  type=str, default=os.path.join(BASE_DIR, 'Datasets', 'UIEB', 'test_raw'))
parser.add_argument('--label_test', type=str, default=os.path.join(BASE_DIR, 'Datasets', 'UIEB', 'test_reference'))
parser.add_argument('--save_folder', type=str, default=os.path.join(BASE_DIR, 'Training_Results_UIEB'))


opt = parser.parse_args()

# Create results folder automatically
os.makedirs(opt.save_folder, exist_ok=True)
CHECKPOINT_INTERVAL = 300

# Print all paths to confirm
print("📁 Dataset and Output Paths:")
print(f"   Train input:  {opt.data_train}")
print(f"   Train labels: {opt.label_train}")
print(f"   Test input:   {opt.data_test}")
print(f"   Test labels:  {opt.label_test}")
print(f"   Results will be saved to: {opt.save_folder}")

# ======================
# Reproducibility
# ======================
def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_torch()
cudnn.benchmark = True

mse_loss = torch.nn.MSELoss().cuda()

# ======================
# Helper: save & load full training state
# ======================
def save_training_state(epoch, iteration, sampler, model, optimizer, path):
    state = {
        "epoch": epoch,
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_indices": sampler.indices if hasattr(sampler, 'indices') else None
    }
    torch.save(state, path)
    print(f"💾 Saved training state → {os.path.basename(path)}")

def load_training_state(model, optimizer, path):
    print(f"📂 Loading checkpoint: {path}")
    state = torch.load(path, map_location='cuda')
    # ---- backward compatibility ----
    if isinstance(state, dict) and "model_state" not in state:
        try:
            model.load_state_dict(state)
            print(f"🔁 Loaded legacy model checkpoint: {os.path.basename(path)}")
        except Exception as e:
            print(f"⚠️ Legacy load failed: {e}")
        return 1, 0, None

    # ---- full-state checkpoint ----
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    epoch = state.get("epoch", 1)
    iteration = state.get("iteration", 0)
    indices = state.get("sampler_indices", None)
    print(f"🔁 Resumed model from {os.path.basename(path)} (epoch={epoch}, iteration={iteration})")
    return epoch, iteration, indices

# ======================
# Training function
# ======================
def train(start_iter_in_epoch=0):
    epoch_loss = 0
    model.train()
    total_batches = len(training_data_loader)
    iteration_offset = start_iter_in_epoch  # global offset if resuming

    print(f"🧮 Total iterations this epoch: {total_batches}, "
          f"resuming from iteration {start_iter_in_epoch + 1}")

    for iteration, batch in enumerate(training_data_loader, 1):
        # Compute true global iteration (clamped to dataset length)
        global_iter = iteration + iteration_offset
        if global_iter > len(train_set):   # stop exactly at dataset end
            print(f"✅ Reached dataset end at iteration {len(train_set)}.")
            break

        input, label = batch[0].cuda(), batch[1].cuda()
        t0 = time.time()

        # Forward + loss
        j_out, t_out, tb_out, a_out = model(input)
        # a_out = get_A(...)  # ❌ replaced by model’s learnable GBL output
        I_rec = j_out * t_out + (1 - tb_out) * a_out

        # --- Adaptive Weighted Smooth L1 Loss ---
        Edge = EdgeLoss()
        # inside training loop, before creating adaptive_loss
        if epoch <= 10:
            current_alpha = 3.0
        else:
            current_alpha = min(5.0, 3.0 + 0.2 * (epoch - 10))

        adaptive_loss = AdaptiveWeightedLoss(alpha=current_alpha, lambda_edge=0.05).cuda()


        # Compute individual components
        l_smooth = F.smooth_l1_loss(j_out, label, beta=1.0)
        l_ssim = 1 - ssim(j_out, label)
        l_edge = Edge(label, j_out)
        l_uiqm = 1 / getUIQM((j_out[0].detach().cpu().permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8))
        l_smooth_r = F.smooth_l1_loss(I_rec, input, beta=1.0)
        l_ssim_r = 1 - ssim(I_rec, input)

        # Combine adaptively (λ for edge fixed)
        loss, adaptive_weights = adaptive_loss([l_smooth, l_ssim, l_edge, l_uiqm, l_smooth_r, l_ssim_r])

        # ✅ GBL Regularization: keep A close to mean scene brightness
        gbl_reg = F.mse_loss(a_out, input.mean(dim=[2,3], keepdim=True))
        loss += 0.01 * gbl_reg  # weighted stabilizer term

        print(f"Adaptive Weights → {[round(w.item(), 4) for w in adaptive_weights]}")



        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # 💾 Save checkpoint every 300 global iterations
        if global_iter % CHECKPOINT_INTERVAL == 0:
            temp_path = os.path.join(opt.save_folder, f"temp_epoch{epoch}_iter{global_iter}.pth")
            save_training_state(epoch, global_iter, train_sampler, model, optimizer, temp_path)
            print(f"💾 Saved checkpoint at global iteration {global_iter}")

        t1 = time.time()
        print(f"===> Epoch[{epoch}]({global_iter}/{len(train_set)}): "
              f"Loss={loss.item():.4f} || Time={(t1 - t0):.2f}s")

    # Average loss for this epoch
    avg_loss = epoch_loss / min(total_batches, len(train_set) - start_iter_in_epoch)
    print(f"🏁 Epoch {epoch} completed — Avg Loss: {avg_loss:.6f}")

    # ✅ Data integrity check (add here)
    if global_iter < len(train_set):
        print(f"⚠️ Warning: Only processed {global_iter} samples this epoch, expected {len(train_set)}")
    else:
        print("✅ Verified: All training samples processed exactly once.")

    return avg_loss


# ======================
# Evaluation
# ======================
# ======================
# Evaluation + Validation Loss
# ======================
def evaluate_model(model, data_path_inp, data_path_gt, save_folder, epoch, train_loss_value=None):
    print(f"\n🔍 Starting OOM-safe validation for Epoch {epoch} ...")
    eval_start = time.time()

    model.eval()
    torch.cuda.empty_cache()
    test_set = get_eval_set(data_path_inp, data_path_gt)
    test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)

    mse_loss_fn = torch.nn.MSELoss().cuda()
    Edge = EdgeLoss()

    psnr_vals, ssim_vals, uiqm_vals, uciqe_vals, mse_vals, val_losses = [], [], [], [], [], []
    skipped = 0

    save_dir = os.path.join(save_folder, f"epoch_{epoch}_samples")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (input_img, target_img, filename) in enumerate(test_loader):
            try:
                input_img = input_img.cuda(non_blocking=True)
                target_img = target_img.cuda(non_blocking=True)

                # forward pass
                j_out, t_out, tb_out, a_out = model(input_img)
        # a_out = get_A(...)  # ❌ replaced by model’s learnable GBL output
                I_rec = j_out * t_out + (1 - tb_out) * a_out

                # validation loss (same as training)
                val_loss = (
                    mse_loss_fn(I_rec, input_img)
                    + mse_loss_fn(target_img, j_out)
                    + 0.05 * Edge(target_img, j_out)
                )
                val_losses.append(val_loss.item())

                # compute metrics
                pred_np = j_out[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)
                gt_np = target_img[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)
                psnr_vals.append(psnr(gt_np, pred_np, data_range=1.0))
                ssim_vals.append(ssim(j_out, target_img).mean().item())
                uiqm_vals.append(getUIQM((pred_np * 255).astype(np.uint8)))
                uciqe_vals.append(getUCIQE((pred_np * 255).astype(np.uint8)))
                mse_vals.append(F.mse_loss(j_out, target_img).item())

                print(f"[{i+1}/{len(test_loader)}] {filename[0]} → ValLoss={val_loss.item():.4f}, PSNR={psnr_vals[-1]:.2f}")

                # === Save raw | enhanced | reference comparison ===
                if i < min(100, len(test_loader)):  # up to 100 or all if fewer
                    import cv2
                    raw_path = os.path.join(data_path_inp, filename[0])
                    ref_path = os.path.join(data_path_gt, filename[0])
                    raw_img = cv2.imread(raw_path)
                    ref_img = cv2.imread(ref_path)
                    if raw_img is not None and ref_img is not None:
                        enh_img = (pred_np * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR

                        # resize to match
                        h = min(raw_img.shape[0], enh_img.shape[0], ref_img.shape[0])
                        w = min(raw_img.shape[1], enh_img.shape[1], ref_img.shape[1])
                        raw_img = cv2.resize(raw_img, (w, h))
                        enh_img = cv2.resize(enh_img, (w, h))
                        ref_img = cv2.resize(ref_img, (w, h))

                        # concatenate horizontally: raw | enhanced | reference
                        combined = np.concatenate((raw_img, enh_img, ref_img), axis=1)

                        # label bar on top
                        label_bar = np.ones((40, combined.shape[1], 3), dtype=np.uint8) * 255
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(label_bar, "RAW", (int(w * 0.15), 30), font, 0.8, (0, 0, 0), 2)
                        cv2.putText(label_bar, "ENHANCED", (int(w * 1.05), 30), font, 0.8, (0, 0, 0), 2)
                        cv2.putText(label_bar, "REFERENCE", (int(w * 2.0), 30), font, 0.8, (0, 0, 0), 2)
                        cv2.putText(label_bar, f"Epoch {epoch} | PSNR: {psnr_vals[-1]:.2f}",
                                    (10, 30), font, 0.7, (100, 0, 0), 2)

                        final_img = np.vstack((label_bar, combined))
                        save_path = os.path.join(save_dir, f"{os.path.splitext(filename[0])[0]}_cmp.png")
                        cv2.imwrite(save_path, final_img)

                # cleanup
                del j_out, t_out, tb_out, I_rec, pred_np, gt_np, val_loss
                torch.cuda.empty_cache()
                gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    skipped += 1
                    print(f"⚠️ Skipping {filename[0]} due to CUDA OOM.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    # === averages ===
    avg_val_loss = np.mean(val_losses) if val_losses else 0
    avg_psnr, avg_ssim = np.mean(psnr_vals), np.mean(ssim_vals)
    avg_uiqm, avg_uciqe = np.mean(uiqm_vals), np.mean(uciqe_vals)
    avg_mse = np.mean(mse_vals)

    print(f"\n📈 Epoch {epoch} Summary → "
          f"TrainLoss={train_loss_value:.6f} | ValLoss={avg_val_loss:.6f} | "
          f"PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f} | "
          f"UIQM={avg_uiqm:.2f} | UCIQE={avg_uciqe:.2f} | MSE={avg_mse:.6f}")
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} samples due to memory limits.")

    # === Save metrics to CSV (with train loss) ===
    csv_path = os.path.join(save_folder, "metrics_log.csv")
    df = pd.DataFrame([{
        "Epoch": epoch,
        "Train_Loss": train_loss_value,
        "Val_Loss": avg_val_loss,
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "UIQM": avg_uiqm,
        "UCIQE": avg_uciqe,
        "MSE": avg_mse
    }])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    # === Plot all metrics ===
    df_all = pd.read_csv(csv_path)
    metrics = ["Train_Loss", "Val_Loss", "PSNR", "SSIM", "UIQM", "UCIQE", "MSE"]
    for metric in metrics:
        plt.figure(figsize=(7,4))
        plt.plot(df_all["Epoch"], df_all[metric], marker='o')
        plt.xlabel("Epoch"); plt.ylabel(metric)
        plt.title(f"{metric} vs Epoch"); plt.grid(True)
        plt.savefig(os.path.join(save_folder, f"{metric}_vs_Epoch.png"), dpi=200)
        plt.close()

    print(f"✅ Evaluation complete for Epoch {epoch}. Took {(time.time()-eval_start)/60:.2f} min.")
    model.train()
    return avg_val_loss



# ======================
# Main
# ======================
if __name__ == '__main__':
    if opt.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found.")

    print('===> Loading dataset')
    train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, opt.data_augmentation)
    train_indices = np.random.permutation(len(train_set))
    train_sampler = SubsetRandomSampler(train_indices)
    training_data_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      sampler=train_sampler, shuffle=False)

    print('===> Building model')
    model = net().cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = lrs.MultiStepLR(optimizer, [opt.decay], opt.gamma)

    # ======== Smart Auto-Resume (epoch or iteration, with sampler) ========
    print("\n🔍 Checking for latest checkpoint...")
    epoch_ckpts = glob.glob(os.path.join(opt.save_folder, "epoch_*.pth"))
    temp_ckpts  = glob.glob(os.path.join(opt.save_folder, "temp_epoch*_iter*.pth"))

    latest_ckpt = None
    if epoch_ckpts or temp_ckpts:
        all_ckpts = [(f, os.path.getmtime(f)) for f in (epoch_ckpts + temp_ckpts)]
        latest_ckpt, _ = max(all_ckpts, key=lambda x: x[1])
        print(f"🧭 Found latest checkpoint: {os.path.basename(latest_ckpt)}")
    else:
        print("🚀 No checkpoint found — starting fresh training.")
        latest_ckpt = None

    start_epoch, start_iter, sampler_indices = 1, 0, None
    if latest_ckpt:
        start_epoch, start_iter, sampler_indices = load_training_state(model, optimizer, latest_ckpt)
        if sampler_indices is not None:
            train_sampler.indices = sampler_indices

        fname = os.path.basename(latest_ckpt)
        if "epoch_" in fname and "iter" not in fname:
            print(f"🔁 Resuming from completed epoch {start_epoch}. Starting epoch {start_epoch + 1}.")
            start_iter = 0
            start_epoch += 1
        else:
            print(f"⏩ Resuming inside epoch {start_epoch} at iteration {start_iter + 1}.")
    else:
        print("🆕 Training from scratch.")

    # ======== Training Loop ========
    # ======== Training Loop ========
import gc

train_loss_list, val_loss_list = [], []

for epoch in range(start_epoch, opt.nEpochs + 1):
    print(f"\n🟢 Epoch {epoch}/{opt.nEpochs}")

    # === TRAIN ===
    torch.cuda.empty_cache()
    gc.collect()

    train_loss = train(start_iter_in_epoch=start_iter if epoch == start_epoch else 0)
    start_iter = 0
    scheduler.step()

    # free memory after training
    torch.cuda.empty_cache()
    gc.collect()

    # === VALIDATE (OOM-safe) ===
    try:
        val_loss = evaluate_model(model, opt.data_test, opt.label_test, opt.save_folder, epoch, train_loss_value=train_loss)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ Validation skipped this epoch due to CUDA OOM.")
            val_loss = np.nan
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise e

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    # === PRINT BOTH LOSSES ===
    print(f"📉 Epoch {epoch} → Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")

    # === SAVE CHECKPOINT ===
    final_path = os.path.join(opt.save_folder, f"epoch_{epoch}.pth")
    save_training_state(epoch, 0, train_sampler, model, optimizer, final_path)

    # cleanup GPU + CPU memory each epoch
    torch.cuda.empty_cache()
    gc.collect()

# ======== SAVE & PLOT LOSS CURVES ========
loss_csv = os.path.join(opt.save_folder, "loss_log.csv")
df_loss = pd.DataFrame({
    "Epoch": np.arange(1, len(train_loss_list)+1),
    "Train_Loss": train_loss_list,
    "Val_Loss": val_loss_list
})
df_loss.to_csv(loss_csv, index=False)

# Plot train loss
plt.figure(figsize=(7,4))
plt.plot(df_loss["Epoch"], df_loss["Train_Loss"], 'b-o')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training Loss vs Epoch"); plt.grid(True)
plt.savefig(os.path.join(opt.save_folder, "Train_Loss_vs_Epoch.png"), dpi=200)
plt.close()

# Plot val loss
plt.figure(figsize=(7,4))
plt.plot(df_loss["Epoch"], df_loss["Val_Loss"], 'r-o')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Validation Loss vs Epoch"); plt.grid(True)
plt.savefig(os.path.join(opt.save_folder, "Val_Loss_vs_Epoch.png"), dpi=200)
plt.close()

print("✅ Training + Validation complete (OOM-safe)!")

