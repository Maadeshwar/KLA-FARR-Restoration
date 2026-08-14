"""
train.py - Training Pipeline for KLA Semiconductor Image Restoration
=====================================================================
Training hygiene checklist:
  - Deterministic seeding (full reproducibility)
  - Train/val split with zero data leakage
  - AMP (Automatic Mixed Precision) for compute efficiency
  - Gradient clipping for stable training
  - Cosine annealing LR schedule
  - Early stopping (patience=20) to prevent overfitting
  - Best model saved on peak validation PSNR
  - All metrics logged to CSV (safe even if Colab crashes)
  - All plots auto-generated and saved to results/ at the end
"""

import os
import sys
import argparse
import random
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SemiconTrainDataset
from src.model import NAFNetSR
from src.loss import CombinedLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim_metric(pred, target, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    ch = pred.size(1)
    kernel = torch.ones(ch, 1, window_size, window_size,
                        device=pred.device, dtype=pred.dtype) / (window_size * window_size)
    mu1 = torch.nn.functional.conv2d(pred, kernel, padding=pad, groups=ch)
    mu2 = torch.nn.functional.conv2d(target, kernel, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = torch.nn.functional.conv2d(pred * pred, kernel, padding=pad, groups=ch) - mu1_sq
    s2 = torch.nn.functional.conv2d(target * target, kernel, padding=pad, groups=ch) - mu2_sq
    s12 = torch.nn.functional.conv2d(pred * target, kernel, padding=pad, groups=ch) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for lr_img, gt_img in loader:
        lr_img = lr_img.to(device, non_blocking=True)
        gt_img = gt_img.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred = model(lr_img)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss, _ = criterion(pred, gt_img)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_psnr, total_ssim, count = 0.0, 0.0, 0.0, 0
    for lr_img, gt_img in loader:
        lr_img = lr_img.to(device, non_blocking=True)
        gt_img = gt_img.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred = model(lr_img)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss, _ = criterion(pred, gt_img)
        total_loss += loss.item()
        pf, gf = pred.float(), gt_img.float()
        total_psnr += compute_psnr(pf, gf)
        total_ssim += compute_ssim_metric(pf, gf)
        count += 1
    n = max(count, 1)
    return total_loss / n, total_psnr / n, total_ssim / n


def generate_all_plots(log_file, results_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available. Skipping plots.")
        return

    os.makedirs(results_dir, exist_ok=True)
    epochs, train_loss, val_loss, psnr, ssim, lr_vals = [], [], [], [], [], []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_loss.append(float(row['Train_Loss']))
            val_loss.append(float(row['Val_Loss']))
            psnr.append(float(row['PSNR']))
            ssim.append(float(row['SSIM']))
            lr_vals.append(float(row['LR']))

    if not epochs:
        print("No epochs logged yet.")
        return

    style = {
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9', 'xtick.color': '#8b949e',
        'ytick.color': '#8b949e', 'grid.color': '#21262d',
        'grid.linewidth': 0.8, 'legend.facecolor': '#21262d',
        'legend.edgecolor': '#30363d',
    }
    plt.rcParams.update(style)

    best_psnr_val = max(psnr)
    best_psnr_ep = epochs[psnr.index(best_psnr_val)]
    best_ssim_val = max(ssim)
    best_ssim_ep = epochs[ssim.index(best_ssim_val)]
    best_val_loss_ep = epochs[val_loss.index(min(val_loss))]

    # Plot 1: Loss Curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, color='#58a6ff', linewidth=2, label='Train Loss')
    ax.plot(epochs, val_loss, color='#f78166', linewidth=2, label='Val Loss', linestyle='--')
    ax.axvline(x=best_val_loss_ep, color='#3fb950', linewidth=1.5, linestyle=':', label=f'Best Epoch ({best_val_loss_ep})')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', color='#e6edf3')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: loss_curve.png")

    # Plot 2: PSNR
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, psnr, color='#3fb950', linewidth=2, label='Val PSNR')
    ax.fill_between(epochs, psnr, alpha=0.15, color='#3fb950')
    ax.axhline(y=best_psnr_val, color='#d29922', linewidth=1.5, linestyle=':', label=f'Peak: {best_psnr_val:.2f} dB (Ep {best_psnr_ep})')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR over Training', fontsize=14, fontweight='bold', color='#e6edf3')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'psnr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: psnr_curve.png")

    # Plot 3: SSIM
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, ssim, color='#d2a8ff', linewidth=2, label='Val SSIM')
    ax.fill_between(epochs, ssim, alpha=0.15, color='#d2a8ff')
    ax.axhline(y=best_ssim_val, color='#d29922', linewidth=1.5, linestyle=':', label=f'Peak: {best_ssim_val:.4f} (Ep {best_ssim_ep})')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM over Training', fontsize=14, fontweight='bold', color='#e6edf3')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'ssim_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: ssim_curve.png")

    # Plot 4: LR Schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lr_vals, color='#ffa657', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Cosine Annealing LR Schedule', fontsize=14, fontweight='bold', color='#e6edf3')
    ax.set_yscale('log'); ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'lr_schedule.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: lr_schedule.png")

    # Plot 5: Combined Dashboard
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    def style_ax(a):
        a.set_facecolor('#161b22')
        a.tick_params(colors='#8b949e')
        a.grid(True, alpha=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color='#58a6ff', linewidth=2, label='Train')
    ax1.plot(epochs, val_loss, color='#f78166', linewidth=2, label='Val', linestyle='--')
    ax1.set_title('Loss Curve', fontweight='bold', color='#e6edf3')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); style_ax(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, psnr, color='#3fb950', linewidth=2)
    ax2.fill_between(epochs, psnr, alpha=0.15, color='#3fb950')
    ax2.axhline(y=best_psnr_val, color='#d29922', linewidth=1.5, linestyle=':', label=f'Peak: {best_psnr_val:.2f} dB')
    ax2.set_title('PSNR (dB)', fontweight='bold', color='#e6edf3')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('dB'); ax2.legend(); style_ax(ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, ssim, color='#d2a8ff', linewidth=2)
    ax3.fill_between(epochs, ssim, alpha=0.15, color='#d2a8ff')
    ax3.axhline(y=best_ssim_val, color='#d29922', linewidth=1.5, linestyle=':', label=f'Peak: {best_ssim_val:.4f}')
    ax3.set_title('SSIM', fontweight='bold', color='#e6edf3')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('SSIM'); ax3.legend(); style_ax(ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, lr_vals, color='#ffa657', linewidth=2)
    ax4.set_title('Learning Rate Schedule', fontweight='bold', color='#e6edf3')
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('LR (log)'); ax4.set_yscale('log'); style_ax(ax4)

    fig.suptitle('KLA Semiconductor Restoration  Training Dashboard\nNAFNet-SR | 4.26M Parameters | Charbonnier + SSIM + Focal Frequency Loss',
                 fontsize=14, fontweight='bold', color='#e6edf3', y=1.01)
    fig.savefig(os.path.join(results_dir, 'training_dashboard.png'), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print("  Saved: training_dashboard.png")
    print(f"\nAll 5 plots saved to: {results_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str,   default='Dataset/train/train')
    parser.add_argument('--save_dir',    type=str,   default='checkpoints')
    parser.add_argument('--results_dir', type=str,   default='results')
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--patience',    type=int,   default=20,
                        help='Early stopping patience. 0 = disabled.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_ds = SemiconTrainDataset(args.data_dir, mode='train', val_split=0.1, seed=args.seed)
    val_ds   = SemiconTrainDataset(args.data_dir, mode='val',   val_split=0.1, seed=args.seed)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = NAFNetSR(img_channel=1, width=32,
                     enc_blk_nums=[2, 2, 4],
                     middle_blk_num=6,
                     dec_blk_nums=[4, 2, 2],
                     upscale=2).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    criterion = CombinedLoss(char_w=1.0, ssim_w=0.1, freq_w=0.05).to(device)

    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    log_file = os.path.join(args.save_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'PSNR', 'SSIM', 'LR', 'Time_s'])

    best_psnr      = 0.0
    epochs_no_gain = 0
    val_psnr       = 0.0
    val_ssim       = 0.0

    print(f'\n{"="*90}')
    print(f'{"Epoch":>6} | {"Train Loss":>10} | {"Val Loss":>10} | {"PSNR (dB)":>10} | {"SSIM":>8} | {"LR":>10} | {"Time":>7}')
    print(f'{"="*90}')
    print(f'Early stopping: {"enabled (patience=" + str(args.patience) + ")" if args.patience > 0 else "DISABLED"}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss               = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f'{epoch:6d} | {train_loss:10.6f} | {val_loss:10.6f} | {val_psnr:10.2f} | {val_ssim:8.4f} | {lr_now:10.2e} | {elapsed:6.1f}s')

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_psnr, val_ssim, lr_now, elapsed])

        if val_psnr > best_psnr:
            best_psnr      = val_psnr
            epochs_no_gain = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr, 'ssim': val_ssim,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f'       -> New best! PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}  [Saved best_model.pt]')
        else:
            epochs_no_gain += 1
            if args.patience > 0 and epochs_no_gain >= args.patience:
                print(f'\n[Early Stopping] No improvement for {args.patience} epochs.')
                print(f'[Early Stopping] Best PSNR: {best_psnr:.2f} dB. Stopping now.')
                break

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'psnr': val_psnr, 'ssim': val_ssim,
    }, os.path.join(args.save_dir, 'final_model.pt'))

    print(f'\n{"="*90}')
    print(f'Training complete.')
    print(f'Best Validation PSNR : {best_psnr:.2f} dB')
    print(f'Checkpoints          : {args.save_dir}/')
    print(f'Metrics log          : {log_file}')
    print(f'\nGenerating all plots...')
    os.system('pip install matplotlib -q')
    generate_all_plots(log_file, args.results_dir)


if __name__ == '__main__':
    main()
