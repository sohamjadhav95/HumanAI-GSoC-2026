"""
evaluate.py
────────────────────────────────────────────────────────────────
RenAIssance OCR1 – GSoC 2026

Evaluation utilities:
  - CER  (Character Error Rate)
  - WER  (Word Error Rate)
  - Normalised Edit Distance
  - Full ablation table: Baseline → +Weighted CTC → +Beam Search → +LLM
  - Qualitative per-line visualisation
────────────────────────────────────────────────────────────────
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(
    predictions: List[str],
    targets: List[str],
    normalise: bool = True,
) -> float:
    """
    Character Error Rate = edit_distance(pred, target) / len(target)

    Args:
        predictions : list of predicted strings
        targets     : list of ground-truth strings
        normalise   : if True, return value in [0, 1]; else raw ratio

    Returns:
        Mean CER across all samples
    """
    if not predictions:
        return 1.0
    errors, total = 0, 0
    for pred, tgt in zip(predictions, targets):
        errors += _edit_distance(pred, tgt)
        total  += len(tgt)
    return errors / max(total, 1)


def compute_wer(
    predictions: List[str],
    targets: List[str],
) -> float:
    """
    Word Error Rate = edit_distance(pred_words, target_words) / len(target_words)
    """
    if not predictions:
        return 1.0
    errors, total = 0, 0
    for pred, tgt in zip(predictions, targets):
        pred_words = pred.split()
        tgt_words  = tgt.split()
        errors += _edit_distance(pred_words, tgt_words)  # type: ignore[arg-type]
        total  += len(tgt_words)
    return errors / max(total, 1)


def compute_ned(predictions: List[str], targets: List[str]) -> float:
    """
    Normalised Edit Distance = edit_distance / max(len_pred, len_target)
    Scale-invariant; useful for comparing lines of different lengths.
    """
    if not predictions:
        return 1.0
    scores = []
    for pred, tgt in zip(predictions, targets):
        denom = max(len(pred), len(tgt), 1)
        scores.append(_edit_distance(pred, tgt) / denom)
    return float(np.mean(scores))


def per_sample_cer(predictions: List[str], targets: List[str]) -> List[float]:
    """Return per-sample CER for error analysis."""
    return [
        _edit_distance(p, t) / max(len(t), 1)
        for p, t in zip(predictions, targets)
    ]


# ══════════════════════════════════════════════════════════════
# ABLATION TABLE
# ══════════════════════════════════════════════════════════════

def build_ablation_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a publication-quality ablation table.

    Args:
        results : {
            "Baseline CRNN"          : {"CER": 0.18, "WER": 0.42, "NED": 0.15},
            "+ Weighted CTC"         : {"CER": 0.14, "WER": 0.35, "NED": 0.12},
            "+ Constrained Beam"     : {"CER": 0.11, "WER": 0.28, "NED": 0.09},
            "+ LLM Post-processing"  : {"CER": 0.07, "WER": 0.19, "NED": 0.06},
        }
        save_path : optional .csv output path

    Returns:
        pandas DataFrame
    """
    rows = []
    prev_cer = None
    for method, metrics in results.items():
        row = {"Method": method}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        if prev_cer is not None and "CER" in metrics:
            delta = metrics["CER"] - prev_cer
            row["ΔCER"] = f"{delta:+.4f}"
        else:
            row["ΔCER"] = "—"
        if "CER" in metrics:
            prev_cer = metrics["CER"]
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("ABLATION TABLE — RenAIssance OCR1")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Ablation table saved → {save_path}")

    return df


def plot_ablation_bars(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart visualising CER/WER improvement across ablation stages."""
    methods = list(results.keys())
    cers = [results[m].get("CER", 0) for m in methods]
    wers = [results[m].get("WER", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, cers, width, label="CER", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, wers, width, label="WER", color="#FF5722", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Error Rate (↓ better)")
    ax.set_title("Ablation Study: Incremental Improvements")
    ax.legend()

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════
# QUALITATIVE VISUALISATION
# ══════════════════════════════════════════════════════════════

def visualise_predictions(
    samples: List[Tuple[str, str]],      # [(image_path, ground_truth), ...]
    predictions_stages: Dict[str, List[str]],  # {stage_name: [pred, ...]}
    n: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """
    Show N line images with ground truth and multi-stage predictions.

    Columns: [Image] | GT | Baseline | +Weighted | +Beam | +LLM
    Correct characters shown in green, errors in red.
    """
    n = min(n, len(samples))
    stages = list(predictions_stages.keys())
    n_cols = 2 + len(stages)   # image + GT + one per stage

    fig = plt.figure(figsize=(20, 2.2 * n))
    gs  = gridspec.GridSpec(n, n_cols, figure=fig,
                            hspace=0.05, wspace=0.05)

    col_headers = ["Line Image", "Ground Truth"] + stages
    for col_idx, header in enumerate(col_headers):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 1.0, header,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")

    for row, (img_path, gt) in enumerate(samples[:n]):
        # Image column
        ax_img = fig.add_subplot(gs[row, 0])
        img = cv2.imread(img_path)
        if img is not None:
            ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.axis("off")

        # Ground truth column
        ax_gt = fig.add_subplot(gs[row, 1])
        ax_gt.text(0.05, 0.5, gt[:80], va="center", fontsize=7,
                   color="#1B5E20", transform=ax_gt.transAxes,
                   wrap=True)
        ax_gt.axis("off")

        # Prediction columns
        for col_offset, stage in enumerate(stages):
            pred = predictions_stages[stage][row] if row < len(
                predictions_stages[stage]) else ""
            cer  = _edit_distance(pred, gt) / max(len(gt), 1)
            color = "#B71C1C" if cer > 0.1 else (
                    "#E65100" if cer > 0.05 else "#2E7D32")
            ax_p = fig.add_subplot(gs[row, 2 + col_offset])
            ax_p.text(0.05, 0.5,
                      f"{pred[:80]}\n[CER={cer:.3f}]",
                      va="center", fontsize=7,
                      color=color,
                      transform=ax_p.transAxes,
                      wrap=True)
            ax_p.axis("off")

    plt.suptitle("Qualitative OCR Results — RenAIssance", y=1.01,
                 fontsize=12, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════

def character_error_breakdown(
    predictions: List[str],
    targets: List[str],
    vocab: Dict[str, int],
) -> pd.DataFrame:
    """
    Per-character error analysis: how often each character is
    substituted, deleted, or inserted.
    Especially useful to show rare-char improvement from weighted CTC.
    """
    from collections import defaultdict
    errors = defaultdict(int)
    totals = defaultdict(int)

    for pred, tgt in zip(predictions, targets):
        # Simple character substitution analysis
        for char in tgt:
            totals[char] += 1
        # Count exact matches
        for pc, tc in zip(pred, tgt):
            if pc != tc:
                errors[tc] += 1

    rows = []
    for char in sorted(totals.keys()):
        err   = errors.get(char, 0)
        total = totals[char]
        rows.append({
            "Character":  char,
            "Occurrences": total,
            "Errors":     err,
            "Error Rate": f"{err / max(total, 1):.3f}",
        })

    df = pd.DataFrame(rows).sort_values("Error Rate", ascending=False)
    return df


def print_worst_samples(
    predictions: List[str],
    targets: List[str],
    image_paths: List[str],
    n: int = 10,
) -> None:
    """Print the N worst-CER samples for qualitative inspection."""
    cers = per_sample_cer(predictions, targets)
    ranked = sorted(zip(cers, predictions, targets, image_paths),
                    key=lambda x: -x[0])

    print(f"\n{'='*60}")
    print(f"TOP-{n} WORST PREDICTIONS")
    print(f"{'='*60}")
    for cer, pred, gt, img_path in ranked[:n]:
        print(f"  File : {Path(img_path).name}")
        print(f"  GT   : {gt}")
        print(f"  PRED : {pred}")
        print(f"  CER  : {cer:.4f}")
        print()
