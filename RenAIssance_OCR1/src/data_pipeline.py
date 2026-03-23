"""
data_pipeline.py
────────────────────────────────────────────────────────────────
RenAIssance OCR1 – GSoC 2026

Stage 1 : PDF  →  JPEG pages          (pdf_to_pages)
Stage 2 : JPEG pages  →  line crops   (extract_lines)
Stage 3 : Ground-truth loading        (load_ground_truth)
Stage 4 : PyTorch Dataset             (OCRDataset)

All outputs are written to the directory structure:
  data/pages/      ← full-page JPEGs
  data/lines/      ← cropped line images
  data/ground_truth/ ← .txt transcription files
────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# STAGE 1 – PDF → JPEG pages
# ══════════════════════════════════════════════════════════════

def pdf_to_pages(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    fmt: str = "JPEG",
) -> List[str]:
    """
    Convert every page of a PDF to a high-resolution JPEG.

    Args:
        pdf_path   : path to the source PDF
        output_dir : directory to save page images
        dpi        : resolution (300 recommended for historical docs)
        fmt        : output format ('JPEG' or 'PNG')

    Returns:
        List of saved image file paths (sorted by page number)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("Install pdf2image:  pip install pdf2image")

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    ext = "jpg" if fmt == "JPEG" else "png"

    logger.info(f"Converting '{pdf_path.name}'  →  {output_dir}  @ {dpi} DPI")
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    saved = []
    for i, page in enumerate(pages, start=1):
        out_path = output_dir / f"{stem}_page_{i:04d}.{ext}"
        page.save(str(out_path), fmt)
        saved.append(str(out_path))

    logger.info(f"  Saved {len(saved)} pages from '{pdf_path.name}'")
    return saved


def batch_pdf_to_pages(
    pdf_dir: str,
    output_dir: str,
    dpi: int = 300,
) -> Dict[str, List[str]]:
    """Convert all PDFs in a directory. Returns {stem: [page_paths]}."""
    pdf_dir = Path(pdf_dir)
    results = {}
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {pdf_dir}")
        return results
    for pdf in pdfs:
        results[pdf.stem] = pdf_to_pages(str(pdf), output_dir, dpi=dpi)
    return results


# ══════════════════════════════════════════════════════════════
# STAGE 2 – Page images → Line crops
# ══════════════════════════════════════════════════════════════

def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu binarization with optional pre-denoising."""
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _remove_borders(binary: np.ndarray, margin: int = 40) -> np.ndarray:
    """Zero out a margin band to remove page-edge noise."""
    out = binary.copy()
    out[:margin, :]  = 0
    out[-margin:, :] = 0
    out[:, :margin]  = 0
    out[:, -margin:] = 0
    return out


def _horizontal_projection(binary: np.ndarray) -> np.ndarray:
    """Row-wise sum of foreground pixels (horizontal projection profile)."""
    return binary.sum(axis=1).astype(np.float32)


def _find_line_bounds(
    projection: np.ndarray,
    min_gap: int = 8,
    min_line_height: int = 15,
    threshold_ratio: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    Identify text-line boundaries from the horizontal projection profile.

    Args:
        projection        : 1-D array of row pixel sums
        min_gap           : minimum blank rows between lines
        min_line_height   : discard lines shorter than this (noise)
        threshold_ratio   : fraction of max projection to treat as blank

    Returns:
        List of (row_start, row_end) tuples
    """
    threshold = projection.max() * threshold_ratio
    in_line = False
    start = 0
    lines = []
    blank_count = 0

    for i, val in enumerate(projection):
        if val > threshold:
            if not in_line:
                in_line = True
                start = i
            blank_count = 0
        else:
            if in_line:
                blank_count += 1
                if blank_count >= min_gap:
                    end = i - blank_count
                    if (end - start) >= min_line_height:
                        lines.append((start, end))
                    in_line = False
                    blank_count = 0

    if in_line:
        lines.append((start, len(projection) - 1))

    return lines


def _find_text_column(
    binary: np.ndarray,
    margin_ratio: float = 0.05,
) -> Tuple[int, int]:
    """
    Find the main text column by vertical projection, ignoring
    extreme left/right margins (handles marginalia in historical docs).

    Returns: (col_left, col_right)
    """
    h, w = binary.shape
    margin = int(w * margin_ratio)
    col_sum = binary[:, margin : w - margin].sum(axis=0)

    # Rolling mean to smooth
    kernel = np.ones(20) / 20
    smoothed = np.convolve(col_sum, kernel, mode="same")
    threshold = smoothed.max() * 0.05

    active = np.where(smoothed > threshold)[0]
    if len(active) == 0:
        return margin, w - margin

    col_left  = max(active[0]  + margin - 10, 0)
    col_right = min(active[-1] + margin + 10, w)
    return col_left, col_right


def extract_lines(
    image_path: str,
    output_dir: str,
    padding_v: int = 4,
    padding_h: int = 8,
    min_width: int = 100,
) -> List[str]:
    """
    Extract individual text-line images from a page image using
    classical horizontal-projection segmentation.

    Historical printed sources: marginalia is suppressed by
    restricting the crop to the main text column.

    Args:
        image_path : path to the page JPEG
        output_dir : directory to save line crops
        padding_v  : vertical padding around each line (pixels)
        padding_h  : horizontal padding (pixels)
        min_width  : discard lines narrower than this (stray marks)

    Returns:
        List of saved line-image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Cannot read image: {image_path}")
        return []

    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    binary = _remove_borders(binary)

    col_l, col_r = _find_text_column(binary)

    # restrict projection to text column only (ignore marginalia)
    col_binary = binary[:, col_l:col_r]
    projection = _horizontal_projection(col_binary)
    line_bounds = _find_line_bounds(projection)

    stem = Path(image_path).stem
    saved = []

    for idx, (r0, r1) in enumerate(line_bounds):
        y0 = max(r0 - padding_v, 0)
        y1 = min(r1 + padding_v, img.shape[0])
        x0 = max(col_l - padding_h, 0)
        x1 = min(col_r + padding_h, img.shape[1])

        crop = img[y0:y1, x0:x1]
        if crop.shape[1] < min_width:
            continue

        out_path = output_dir / f"{stem}_line_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved.append(str(out_path))

    logger.info(f"  {Path(image_path).name}  →  {len(saved)} lines")
    return saved



def split_spread_page(image_path: str, output_dir: str) -> tuple:
    """
    Split a two-page spread scan into left / right individual pages.
    CRITICAL for Buendia: each PDF page is a spread showing BOTH book
    pages side-by-side. The gutter (binding valley) has very low ink
    density and is detected via the vertical projection profile.
    Returns (left_path, right_path), or (original_path, None) on failure.
    """
    import shutil
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Cannot read: {image_path}")
        return None, None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    s, e = w // 3, 2 * w // 3
    col_sum  = binary[:, s:e].sum(axis=0).astype(float)
    kernel_n = max(5, w // 100)
    smoothed = np.convolve(col_sum, np.ones(kernel_n) / kernel_n, mode="same")
    gutter_local = int(smoothed.argmin())
    gutter_x = gutter_local + s
    margin   = max(8, int(w * 0.01))
    left_x2  = gutter_x - margin
    right_x1 = gutter_x + margin
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if left_x2 < w // 8 or right_x1 > 7 * w // 8:
        dst = out_dir / Path(image_path).name
        shutil.copy(str(image_path), str(dst))
        return str(dst), None
    stem = Path(image_path).stem
    left_path  = str(out_dir / f"{stem}_L.jpg")
    right_path = str(out_dir / f"{stem}_R.jpg")
    cv2.imwrite(left_path,  img[:, :left_x2])
    cv2.imwrite(right_path, img[:, right_x1:])
    return left_path, right_path


def batch_split_spreads(pages_dir: str, split_dir: str) -> List[str]:
    """
    Split all two-page spread images into individual pages.
    Automatically detects spreads vs. single pages using the vertical
    projection gutter test. Returns sorted list of all individual pages.
    """
    import shutil
    pages_dir = Path(pages_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    all_pages = []
    images = sorted(pages_dir.glob("*.jpg")) + sorted(pages_dir.glob("*.png"))
    for img_path in tqdm(images, desc="Splitting spreads"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        s, e = w // 3, 2 * w // 3
        col_sum  = binary[:, s:e].sum(axis=0).astype(float)
        col_mean = col_sum.mean() + 1e-6
        is_spread = (col_sum.min() / col_mean) < 0.15
        if is_spread:
            l, r = split_spread_page(str(img_path), str(split_dir))
            if l: all_pages.append(l)
            if r: all_pages.append(r)
        else:
            dst = split_dir / img_path.name
            shutil.copy(str(img_path), str(dst))
            all_pages.append(str(dst))
    logger.info(f"Split {len(images)} pages → {len(all_pages)} individual pages")
    return sorted(all_pages)


def batch_extract_lines(
    pages_dir: str,
    lines_dir: str,
) -> Dict[str, List[str]]:
    """Extract lines from all page images in a directory."""
    pages_dir = Path(pages_dir)
    results = {}
    images = sorted(pages_dir.glob("*.jpg")) + sorted(pages_dir.glob("*.png"))
    for img_path in tqdm(images, desc="Extracting lines"):
        lines = extract_lines(str(img_path), lines_dir)
        results[img_path.stem] = lines
    return results


# ══════════════════════════════════════════════════════════════
# STAGE 3 – Ground-truth loading
# ══════════════════════════════════════════════════════════════

def load_ground_truth(gt_dir: str) -> Dict[str, List[str]]:
    """
    Load ground-truth transcriptions from .txt files.

    Each .txt file corresponds to one source (same stem as the PDF).
    Lines in the .txt file map one-to-one to the extracted line images.

    Returns:
        {source_stem: [line1_text, line2_text, ...]}
    """
    gt_dir = Path(gt_dir)
    gt = {}
    for txt_file in sorted(gt_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f.readlines() if l.strip()]
        gt[txt_file.stem] = lines
        logger.info(f"  Loaded GT '{txt_file.name}' → {len(lines)} lines")
    return gt


def build_char_vocab(gt_dict: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Build character-to-index vocabulary from ground-truth texts.
    Index 0 is reserved for CTC blank token.

    Returns:
        {char: index}  (blank = 0, characters start from 1)
    """
    chars = set()
    for lines in gt_dict.values():
        for line in lines:
            chars.update(line)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<blank>"] = 0
    logger.info(f"Vocabulary size: {len(vocab)} characters (incl. blank)")
    return vocab


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_char_frequencies(gt_dict: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute normalised inverse-frequency weights per character.
    Rare characters (e.g. ſ, ã, ligatures) get higher weight.

    Returns:
        {char: weight}  — higher weight → rarer character
    """
    from collections import Counter
    counts: Counter = Counter()
    for lines in gt_dict.values():
        for line in lines:
            counts.update(line)

    total = sum(counts.values())
    # Inverse frequency, clamped to [1.0, 10.0]
    weights = {}
    for char, count in counts.items():
        freq = count / total
        w = min(max(1.0 / (freq * len(counts)), 1.0), 10.0)
        weights[char] = round(w, 4)

    logger.info(f"Computed char weights. Top-5 rarest: "
                f"{sorted(weights.items(), key=lambda x: -x[1])[:5]}")
    return weights


# ══════════════════════════════════════════════════════════════
# STAGE 4 – PyTorch Dataset
# ══════════════════════════════════════════════════════════════

class OCRDataset(Dataset):
    """
    PyTorch Dataset for line-level OCR training.

    Each sample is (image_tensor, label_tensor, label_length).
    Images are resized to a fixed height; width varies (handled by
    the DataLoader collate function below).
    """

    IMG_HEIGHT = 64  # fixed height for CRNN input

    def __init__(
        self,
        samples: List[Tuple[str, str]],   # [(image_path, text), ...]
        vocab: Dict[str, int],
        augment: bool = False,
        max_width: int = 2048,
    ):
        self.samples   = samples
        self.vocab     = vocab
        self.augment   = augment
        self.max_width = max_width
        self._build_transforms()

    def _build_transforms(self):
        basic = [
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
        if self.augment:
            aug = [
                T.RandomApply([T.GaussianBlur(3)], p=0.3),
                T.RandomAdjustSharpness(2, p=0.3),
                T.RandomAutocontrast(p=0.3),
                T.ColorJitter(brightness=0.3, contrast=0.3),
            ]
            self.transform = T.Compose(aug + basic)
        else:
            self.transform = T.Compose(basic)

    def _resize_keep_aspect(self, img: Image.Image) -> Image.Image:
        """Resize to fixed height, keep aspect ratio, cap width."""
        w, h = img.size
        new_w = min(int(w * self.IMG_HEIGHT / h), self.max_width)
        return img.resize((new_w, self.IMG_HEIGHT), Image.LANCZOS)

    def _encode_text(self, text: str) -> torch.Tensor:
        indices = [self.vocab[c] for c in text if c in self.vocab]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._resize_keep_aspect(img)
        img_tensor = self.transform(img)          # (1, H, W)
        label      = self._encode_text(text)
        return img_tensor, label, torch.tensor(len(label), dtype=torch.long)


def collate_fn(batch):
    """
    Pad images in a batch to the same width (max in batch).
    Returns:
        images  : (B, 1, H, W_max)
        labels  : (sum_of_label_lengths,)  – flat for CTC
        lengths : (B,)                     – label lengths
        widths  : (B,)                     – input lengths for CTC
    """
    images, labels, lengths = zip(*batch)

    max_w = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        pad_w = max_w - img.shape[2]
        padded.append(torch.nn.functional.pad(img, (0, pad_w), value=-1.0))

    images_tensor  = torch.stack(padded)            # (B, 1, H, W)
    labels_flat    = torch.cat(labels)              # (sum_lens,)
    lengths_tensor = torch.stack(lengths)           # (B,)

    # CTC input lengths = W dimension after CNN+pool (computed in model)
    # We pass the raw pixel widths; model will down-sample by factor 4
    widths = torch.tensor(
        [img.shape[2] for img in images], dtype=torch.long
    )

    return images_tensor, labels_flat, lengths_tensor, widths


def build_samples_from_lines(
    lines_dir: str,
    gt_dict: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """
    Match extracted line images to ground-truth text lines.

    Convention: line image files are named   {source_stem}_page_XXXX_line_YYYY.jpg
    and ground-truth .txt files are named   {source_stem}.txt
    where each line in the .txt corresponds to consecutive line images
    from that source, in sorted order.

    Returns:
        [(image_path, text), ...]
    """
    lines_dir = Path(lines_dir)
    samples = []

    for source_stem, gt_lines in gt_dict.items():
        # Collect all line images for this source (sorted)
        pattern = f"{source_stem}*line*.jpg"
        img_paths = sorted(lines_dir.glob(pattern))

        # Pair images with GT lines (take the min to avoid IndexError)
        n = min(len(img_paths), len(gt_lines))
        if n == 0:
            logger.warning(f"No matching line images for source '{source_stem}'")
            continue

        for img_path, text in zip(img_paths[:n], gt_lines[:n]):
            if text.strip():   # skip empty GT lines
                samples.append((str(img_path), text.strip()))

        logger.info(f"  '{source_stem}': {n} paired samples")

    logger.info(f"Total samples: {len(samples)}")
    return samples


# ══════════════════════════════════════════════════════════════
# QUICK DIAGNOSTICS
# ══════════════════════════════════════════════════════════════

def visualize_line_extraction(
    image_path: str,
    n_lines: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Display first N extracted lines from a page (for notebook QA)."""
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    binary = _remove_borders(binary)
    col_l, col_r = _find_text_column(binary)
    projection   = _horizontal_projection(binary[:, col_l:col_r])
    line_bounds  = _find_line_bounds(projection)

    fig, axes = plt.subplots(min(n_lines, len(line_bounds)), 1,
                             figsize=(14, 2 * min(n_lines, len(line_bounds))))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (r0, r1) in zip(axes, line_bounds[:n_lines]):
        crop = img[r0:r1, col_l:col_r]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    plt.suptitle(f"Line extraction preview: {Path(image_path).name}", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
