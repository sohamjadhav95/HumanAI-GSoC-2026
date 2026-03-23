"""
train.py
────────────────────────────────────────────────────────────────
RenAIssance OCR1 – GSoC 2026

Training loop:
  - Weighted CTC loss (rare-character upweighting)
  - OneCycleLR scheduler
  - Early stopping on validation CER
  - Greedy-decode monitoring every N steps
  - TensorBoard / matplotlib loss curves
────────────────────────────────────────────────────────────────
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import (
    CRNN, WeightedCTCLoss, greedy_decode,
    save_checkpoint, load_checkpoint, build_model,
)
from data_pipeline import OCRDataset, collate_fn
from evaluate import compute_cer, compute_wer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ══════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Model
    "hidden_size":      256,
    "lstm_layers":      2,
    "dropout":          0.3,
    "pretrained_cnn":   True,

    # Training
    "batch_size":       16,
    "num_epochs":       50,
    "learning_rate":    3e-4,
    "weight_decay":     1e-4,
    "grad_clip":        5.0,
    "val_split":        0.15,

    # Weighted CTC
    "weight_scale":     1.5,     # amplify rare-char penalty

    # Early stopping
    "patience":         8,       # epochs without CER improvement

    # Monitoring
    "log_interval":     50,      # steps between greedy-decode logs
    "save_dir":         "outputs/checkpoints",
}


# ══════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════

def train_epoch(
    model:      CRNN,
    loader:     DataLoader,
    criterion:  WeightedCTCLoss,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    device:     torch.device,
    idx2char:   Dict[int, str],
    grad_clip:  float = 5.0,
    log_interval: int = 50,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for step, (images, labels, label_lengths, pixel_widths) in enumerate(
        tqdm(loader, desc="  train", leave=False)
    ):
        images        = images.to(device)
        labels        = labels.to(device)
        label_lengths = label_lengths.to(device)

        # Forward
        log_probs    = model(images)                        # (T, B, C)
        input_lengths = model.input_lengths(pixel_widths.to(device))

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        # ── Greedy-decode monitoring ─────────────────────────
        if step % log_interval == 0:
            with torch.no_grad():
                preds = greedy_decode(log_probs, input_lengths, idx2char)
                offset = 0
                sample_gt = []
                for l in label_lengths[:3]:
                    n = l.item()
                    sample_gt.append(
                        "".join(idx2char.get(i.item(), "") for i in labels[offset:offset+n])
                    )
                    offset += n
                for pred, gt in zip(preds[:3], sample_gt):
                    logger.debug(f"    GT  : {gt}")
                    logger.debug(f"    PRED: {pred}")

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model:     CRNN,
    loader:    DataLoader,
    criterion: WeightedCTCLoss,
    device:    torch.device,
    idx2char:  Dict[int, str],
) -> Tuple[float, float, float]:
    """
    Returns (val_loss, CER, WER).
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels, label_lengths, pixel_widths in tqdm(
        loader, desc="  val", leave=False
    ):
        images        = images.to(device)
        labels        = labels.to(device)
        label_lengths = label_lengths.to(device)

        log_probs     = model(images)
        input_lengths = model.input_lengths(pixel_widths.to(device))

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        total_loss += loss.item()

        preds = greedy_decode(log_probs, input_lengths, idx2char)
        all_preds.extend(preds)

        offset = 0
        for l in label_lengths:
            n = l.item()
            gt = "".join(idx2char.get(i.item(), "") for i in labels[offset:offset+n])
            all_targets.append(gt)
            offset += n

    cer = compute_cer(all_preds, all_targets)
    wer = compute_wer(all_preds, all_targets)
    val_loss = total_loss / max(len(loader), 1)
    return val_loss, cer, wer


# ══════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════

def train(
    samples:      List[Tuple[str, str]],
    vocab:        Dict[str, int],
    char_weights: Optional[Dict[str, float]] = None,
    config:       Optional[Dict] = None,
    resume_path:  Optional[str]  = None,
) -> Tuple[CRNN, Dict]:
    """
    Full training pipeline.

    Args:
        samples      : [(image_path, text), ...]
        vocab        : {char: index}  (blank = 0)
        char_weights : {char: float}  inverse-frequency weights
        config       : training hyperparameters (see DEFAULT_CONFIG)
        resume_path  : path to checkpoint to resume from

    Returns:
        (best_model, history_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # ── Dataset split ────────────────────────────────────────
    n_val   = max(1, int(len(samples) * cfg["val_split"]))
    n_train = len(samples) - n_val

    dataset = OCRDataset(samples, vocab, augment=False)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Enable augmentation only on training split
    train_ds.dataset.augment = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train: {n_train} | Val: {n_val}")

    # ── Model ────────────────────────────────────────────────
    model = build_model(
        vocab,
        hidden_size=cfg["hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        pretrained_cnn=cfg["pretrained_cnn"],
    ).to(device)

    idx2char = {v: k for k, v in vocab.items() if k != "<blank>"}

    # ── Loss ─────────────────────────────────────────────────
    criterion = WeightedCTCLoss(
        vocab=vocab,
        char_weights=char_weights,
        blank_idx=0,
        weight_scale=cfg["weight_scale"],
    )

    # ── Optimiser & Scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["num_epochs"],
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 0
    best_cer    = float("inf")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_cer = load_checkpoint(
            resume_path, model, optimizer, device=str(device)
        )

    # ── History ──────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_cer":    [],
        "val_wer":    [],
    }

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    patience_counter = 0

    # ══════════════════════════════════════════════════════
    # EPOCH LOOP
    # ══════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg["num_epochs"]):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, idx2char,
            grad_clip=cfg["grad_clip"],
            log_interval=cfg["log_interval"],
        )
        val_loss, cer, wer = validate(
            model, val_loader, criterion, device, idx2char
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:3d}/{cfg['num_epochs']} | "
            f"loss {train_loss:.4f} → {val_loss:.4f} | "
            f"CER {cer:.4f} | WER {wer:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(cer)
        history["val_wer"].append(wer)

        # ── Save best ────────────────────────────────────────
        if cer < best_cer:
            best_cer = cer
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch + 1, cer,
                str(save_dir / "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no CER improvement for {cfg['patience']} epochs)"
                )
                break

        # Always save latest
        save_checkpoint(
            model, optimizer, epoch + 1, cer,
            str(save_dir / "latest_model.pt"),
        )

    logger.info(f"Training complete. Best validation CER: {best_cer:.4f}")

    # ── Load best model before returning ─────────────────────
    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), model, device=str(device))

    return model, history


# ══════════════════════════════════════════════════════════════
# LEARNING CURVE PLOT
# ══════════════════════════════════════════════════════════════

def plot_history(history: Dict, save_path: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train loss", color="#2196F3")
    axes[0].plot(history["val_loss"],   label="Val loss",   color="#FF5722")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    # CER curve
    axes[1].plot(history["val_cer"], color="#4CAF50", linewidth=2)
    axes[1].set_title("Validation CER (↓ better)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("CER")

    # WER curve
    axes[2].plot(history["val_wer"], color="#9C27B0", linewidth=2)
    axes[2].set_title("Validation WER (↓ better)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("WER")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Learning curves saved → {save_path}")
