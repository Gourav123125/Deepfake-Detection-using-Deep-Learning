"""
train.py
========
Training script for the DeepFake Detection System.

Features:
- Mixed precision training (FP16)
- Differential learning rates (backbone vs head)
- OneCycleLR scheduler with warmup
- Early stopping (patience=7) by AUC-ROC
- WeightedRandomSampler for class imbalance
- Full metric logging per epoch

Usage:
    python train.py --data-root /path/to/dataset --epochs 30 --batch 16
    python train.py --data-root /data --epochs 50 --batch 8 --mode video
    python train.py --data-root /data --resume checkpoints/deepfake_best.pth
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from model.architecture import build_model
from utils.dataset import DeepFakeImageDataset, DeepFakeVideoDataset
from utils.transforms import get_transforms, get_video_transforms
from utils.metrics import MetricTracker, plot_training_history


# ─────────────────────────────────────────────────────────────────
#  Trainer Class
# ─────────────────────────────────────────────────────────────────
class Trainer:
    """
    Full training loop with mixed precision, scheduling,
    early stopping, and checkpoint management.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.mode = config.get("mode", "image")   # 'image' or 'video'

        # Device selection: CUDA → MPS → CPU
        self.device = torch.device(
            "cuda"  if torch.cuda.is_available() else
            "mps"   if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"[Trainer] Device: {self.device}")

        # ── Model ────────────────────────────────────────────────
        self.model = build_model(
            pretrained   = config.get("pretrained", True),
            dropout      = config.get("dropout", 0.4),
            lstm_hidden  = config.get("lstm_hidden", 512),
            lstm_layers  = config.get("lstm_layers", 2),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable    = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total_params:,} | Trainable: {trainable:,}")

        # ── Loss ─────────────────────────────────────────────────
        # Upweight fake class: deepfake miss is more costly than false alarm
        fake_weight = config.get("fake_weight", 1.5)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, fake_weight]).to(self.device)
        )

        # ── Optimizer (differential LR) ───────────────────────────
        # Backbone gets lower LR to preserve pretrained features
        lr = config["lr"]
        backbone_params = list(self.model.features.parameters())
        head_params = (
            list(self.model.cbam.parameters()) +
            list(self.model.freq_head.parameters()) +
            list(self.model.lstm.parameters()) +
            list(self.model.classifier_image.parameters()) +
            list(self.model.classifier_video.parameters())
        )
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},  # 10× lower for backbone
            {"params": head_params,     "lr": lr},
        ], weight_decay=config.get("weight_decay", 1e-4))

        # Scheduler set up after dataloader creation
        self.scheduler = None

        # ── Mixed Precision ───────────────────────────────────────
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == "cuda")
        )

        # ── State ─────────────────────────────────────────────────
        self.best_auc = 0.0
        self.patience_counter = 0
        self.history = []

        self.save_dir = Path(config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_scheduler(self, steps_per_epoch: int):
        """Build OneCycleLR after knowing dataset size."""
        total_steps = self.cfg["epochs"] * steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.cfg["lr"] * 0.1, self.cfg["lr"]],
            total_steps=total_steps,
            pct_start=0.05,          # 5% of training for warmup
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )

    # ── One Epoch: Train ─────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        tracker = MetricTracker()

        for batch_idx, batch in enumerate(loader):
            if self.mode == "video":
                imgs, labels = batch
                imgs = imgs.to(self.device)   # (B, T, C, H, W)
            else:
                imgs, labels = batch
                imgs = imgs.to(self.device)   # (B, C, H, W)

            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(imgs, mode=self.mode)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            tracker.update(loss.item(), logits.detach(), labels)

            # Progress log every 50 batches
            if batch_idx % 50 == 0:
                m = tracker.compute()
                lr_now = self.optimizer.param_groups[1]["lr"]
                print(
                    f"  [{batch_idx:4d}/{len(loader)}] "
                    f"Loss: {m['loss']:.4f} | Acc: {m['acc']:.3f} | "
                    f"AUC: {m['auc']:.3f} | LR: {lr_now:.2e}"
                )

        return tracker.compute()

    # ── One Epoch: Validate ──────────────────────────────────────
    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        tracker = MetricTracker()

        for batch in loader:
            imgs, labels = batch
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(imgs, mode=self.mode)
                loss   = self.criterion(logits, labels)

            tracker.update(loss.item(), logits, labels)

        return tracker.compute()

    # ── Checkpoint ───────────────────────────────────────────────
    def save_checkpoint(self, metrics: Dict, epoch: int, tag: str = "best"):
        ckpt = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":         metrics,
            "config":          self.cfg,
        }
        path = self.save_dir / f"deepfake_{tag}.pth"
        torch.save(ckpt, path)
        print(f"  [Checkpoint] Saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_auc = ckpt["metrics"].get("auc", 0.0)
        print(f"  [Resume] Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ── Main Train Loop ──────────────────────────────────────────
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        self._build_scheduler(len(train_loader))
        patience = self.cfg.get("patience", 7)

        for epoch in range(1, self.cfg["epochs"] + 1):
            t0 = time.time()
            print(f"\n{'─'*65}")
            print(f"  Epoch {epoch}/{self.cfg['epochs']}  |  Mode: {self.mode}")
            print(f"{'─'*65}")

            train_m = self.train_epoch(train_loader)
            val_m   = self.val_epoch(val_loader)
            elapsed = time.time() - t0

            # Log
            print(f"\n  Train | Loss: {train_m['loss']:.4f} | Acc: {train_m['acc']:.4f} | "
                  f"AUC: {train_m['auc']:.4f} | F1: {train_m['f1']:.4f}")
            print(f"  Val   | Loss: {val_m['loss']:.4f} | Acc: {val_m['acc']:.4f} | "
                  f"AUC: {val_m['auc']:.4f} | F1: {val_m['f1']:.4f}")
            print(f"  Time:  {elapsed:.1f}s")

            self.history.append({
                "epoch": epoch,
                "train": train_m,
                "val":   val_m,
            })

            # Save best by AUC
            if val_m["auc"] > self.best_auc:
                self.best_auc = val_m["auc"]
                self.save_checkpoint(val_m, epoch, tag="best")
                self.patience_counter = 0
                print(f"  ★ New best AUC: {self.best_auc:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ↓ No improvement ({self.patience_counter}/{patience})")

            # Periodic save every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(val_m, epoch, tag=f"epoch{epoch:03d}")

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n  [Early Stop] Patience exhausted at epoch {epoch}.")
                break

        # Save training history
        hist_path = self.save_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Plot curves
        try:
            plot_training_history(
                self.history,
                str(self.save_dir / "training_curves.png")
            )
        except Exception as e:
            print(f"  [Warning] Could not plot training curves: {e}")

        print(f"\n{'='*65}")
        print(f"  Training complete.  Best Val AUC: {self.best_auc:.4f}")
        print(f"{'='*65}")
        return self.history


# ─────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train DeepFake Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root",  type=str, required=True,
                        help="Dataset root with real/ and fake/ subdirs")
    parser.add_argument("--mode",       type=str, default="image",
                        choices=["image", "video"],
                        help="Training mode: image or video (LSTM)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--save-dir",   type=str,   default="checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--num-frames", type=int,   default=16,
                        help="Frames per video (video mode only)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap samples per class for quick experiments")
    args = parser.parse_args()

    config = {
        "mode":          args.mode,
        "epochs":        args.epochs,
        "lr":            args.lr,
        "batch_size":    args.batch,
        "save_dir":      args.save_dir,
        "pretrained":    True,
        "dropout":       0.4,
        "fake_weight":   1.5,
        "weight_decay":  1e-4,
        "patience":      7,
        "lstm_hidden":   512,
        "lstm_layers":   2,
    }

    # ── Build Dataset ──────────────────────────────────────────
    if args.mode == "image":
        tf_train = get_transforms("train")
        tf_val   = get_transforms("val")
        full_ds  = DeepFakeImageDataset(
            args.data_root,
            transform=None,
            max_per_class=args.max_per_class,
        )
    else:
        tf_train = get_video_transforms("train")
        tf_val   = get_video_transforms("val")
        full_ds  = DeepFakeVideoDataset(
            args.data_root,
            transform=None,
            num_frames=args.num_frames,
        )

    # ── Train / Val Split ──────────────────────────────────────
    n       = len(full_ds)
    n_val   = int(n * args.val_split)
    n_train = n - n_val

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms
    train_ds.dataset.transform = tf_train
    val_ds.dataset.transform   = tf_val

    # ── Balanced Sampler ───────────────────────────────────────
    all_weights   = full_ds.get_sample_weights()
    train_weights = all_weights[train_ds.indices]
    sampler = WeightedRandomSampler(
        train_weights, len(train_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"\n[Data] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Train ──────────────────────────────────────────────────
    trainer = Trainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
