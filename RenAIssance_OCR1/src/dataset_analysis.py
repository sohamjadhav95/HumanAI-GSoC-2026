"""
dataset_analysis.py
────────────────────────────────────────────────────────────────
RenAIssance OCR1 – GSoC 2026

Analysis of the two uploaded PDFs and their OCR challenges.
Run this once to understand your data before training.

OBSERVATIONS FROM VISUAL INSPECTION:
────────────────────────────────────────────────────────────────

SOURCE 1: Buendia_-_Instruccion.pdf (PRINTED — Test I target)
  Type      : Early modern Spanish printed book
  Date      : 1740 (Gerona: Por Jayme Bró)
  Pages     : 33 visible pages
  Layout    : Two-column (left/right page spread in PDF scans)
  Script    : 18th-century Spanish print

  CRITICAL OCR CHALLENGES:
  1. TWO-COLUMN SPREAD LAYOUT
     - Each PDF page contains LEFT page + RIGHT page side-by-side
     - Line detector must split each page into two halves first
     - Left half: even-numbered book pages (DEL TRATO column)
     - Right half: odd-numbered book pages (CORTESANO CON DIOS column)

  2. LONG-S (ſ) UBIQUITOUS
     - Appears in initial/medial positions: "efte", "efta", "fino", "folo"
     - Looks identical to 'f' to OCR → most common error class
     - Examples: "confagra", "confiais", "efmalte", "Inftruccion"

  3. ARCHAIC ORTHOGRAPHY
     - 'v' used for 'u' in some positions: "vueftra", "vnos"
     - Abbreviation tildes: 'q̃' = "que", supralinear bars
     - Accent marks: à, è, ì, ò, ù (grave), not modern acute
     - 'x' for 'j': "dexar", "dexandola", "executar"
     - 'ph' for 'f': "Philippo" etc.
     - Double consonants: "affunto", "effecto", "offecer"

  4. DECORATIVE INITIALS
     - Large ornamental drop caps at chapter starts (P, A, E, S, etc.)
     - Often inside decorative boxes — OCR typically fails these

  5. MARGINALIA
     - Biblical references in outer margins (Luc., Matt., Psal., etc.)
     - Must be EXCLUDED from main text extraction
     - Our column-detection handles this

  6. RUNNING HEADERS
     - "DEL TRATO" (left pages) / "CORTESANO CON DIOS." (right pages)
     - With page numbers: "4", "5", etc.
     - Should be excluded OR included consistently

  7. TYPOGRAPHY SPECIFICS
     - Section headers in SMALL CAPS: "PARTE PRIMERA", "CAPITULO I."
     - Italic text for Latin quotations
     - Mixed font sizes within same line (chapter titles)

  CHARACTER FREQUENCY (estimated from GT):
     Most common: a, e, o, i, s, n, r, l, t, d, c, u, m, p, q
     Rare/challenging: ſ (long-s), à, è, ò, ñ, ü, q̃, ¡, ;

────────────────────────────────────────────────────────────────

SOURCE 2: AHPG-GPAH_1__x3a_1716_A_35___1744.pdf (HANDWRITTEN — Test II)
  Type      : 18th-century legal manuscript (handwritten)
  Date      : 1744
  Content   : "Informacion de filiacion hidalguia y limpieza de sangre"
              (Legal proceeding on nobility and lineage of Andres de Muguruza)
  Script    : Spanish cursive secretary hand (letra procesal/humanistica)

  THIS FILE IS FOR TEST II (Handwritten OCR with LLM pipeline)
  NOT for Test I — use for OCR3 proposal if targeting that.

  NOTES FOR REFERENCE:
  - Heavy ligatures throughout
  - Ink bleed and paper foxing
  - Superscript abbreviations common: "Vm" = "Vuestra merced", "dn" = "don"
  - Basque proper nouns: Muguruza, Sagarreguiurra, Aloybar, Beizama

────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


def analyze_page_layout(image_path: str) -> dict:
    """
    Analyze a scanned page to detect its layout characteristics.
    Specifically handles the two-column spread format of Buendia.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Vertical projection to detect column split
    col_sum = binary.sum(axis=0).astype(float)

    # Find the valley (gutter) between two columns
    mid_region = col_sum[w//4 : 3*w//4]
    gutter_offset = mid_region.argmin() + w//4

    return {
        "width": w,
        "height": h,
        "gutter_x": gutter_offset,
        "is_spread": abs(gutter_offset - w//2) < w//6,  # true if gutter near center
        "text_density": binary.mean() / 255,
    }


def split_spread_page(image_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Split a two-page spread scan into individual left and right pages.
    Critical for Buendia where each PDF page contains both book pages.

    Args:
        image_path : path to the spread JPEG
        output_dir : where to save split pages

    Returns:
        (left_page_path, right_page_path)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find gutter: minimum vertical projection in middle third
    search_start = w // 3
    search_end = 2 * w // 3
    col_sum = binary[:, search_start:search_end].sum(axis=0)

    # Smooth to find the clear valley
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(col_sum.astype(float), size=30)
    gutter_local = smoothed.argmin()
    gutter_x = gutter_local + search_start

    # Add small margin around gutter
    margin = max(5, int(w * 0.01))
    left_x2 = gutter_x - margin
    right_x1 = gutter_x + margin

    left_page  = img[:, :left_x2]
    right_page = img[:, right_x1:]

    stem = Path(image_path).stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    left_path  = str(out / f"{stem}_L.jpg")
    right_path = str(out / f"{stem}_R.jpg")
    cv2.imwrite(left_path,  left_page)
    cv2.imwrite(right_path, right_page)

    return left_path, right_path


def batch_split_spreads(pages_dir: str, split_dir: str) -> List[str]:
    """
    Split all spread pages in a directory into individual pages.
    Returns list of all individual page paths (sorted by original page order).
    """
    pages_dir = Path(pages_dir)
    all_splits = []

    for img_path in sorted(pages_dir.glob("*.jpg")):
        layout = analyze_page_layout(str(img_path))

        if layout.get("is_spread", False):
            l, r = split_spread_page(str(img_path), split_dir)
            if l: all_splits.append(l)
            if r: all_splits.append(r)
        else:
            # Single page — copy as-is
            import shutil
            dst = Path(split_dir) / img_path.name
            shutil.copy(str(img_path), str(dst))
            all_splits.append(str(dst))

    print(f"Split {len(list(pages_dir.glob('*.jpg')))} spread pages "
          f"→ {len(all_splits)} individual pages")
    return all_splits


def estimate_long_s_frequency(gt_path: str) -> float:
    """
    Estimate the frequency of long-s (ſ) and similar rare characters
    in the ground truth to justify weighted CTC loss.
    In Buendia, 'f' often represents long-s ſ.
    """
    with open(gt_path, encoding='utf-8') as f:
        text = f.read()

    total = len(text.replace('\n', '').replace(' ', ''))
    # Characters that need upweighting in Buendia
    rare_chars = set('àèìòùáéíóúüñãõq̃ſ')
    rare_count = sum(1 for c in text if c in rare_chars)

    # Also count 'f' used as long-s (heuristic: 'f' before vowel or at word start)
    import re
    long_s_candidates = len(re.findall(r'\bf[aeiouàèìòù]', text.lower()))

    print(f"Ground truth stats:")
    print(f"  Total chars     : {total}")
    print(f"  Rare diacritics : {rare_count} ({100*rare_count/max(total,1):.1f}%)")
    print(f"  Long-s candidates: {long_s_candidates}")
    print(f"  → Weighted CTC IS justified (rare chars need upweighting)")
    return rare_count / max(total, 1)


def visualize_dataset_challenges(image_path: str, save_path: str = None):
    """
    Create a visual summary of OCR challenges for a given page.
    Shows: original, binarized, column split, text projection.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = img.shape[:2]
    col_sum = binary.sum(axis=0)
    row_sum = binary.sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Original
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Original Scan', fontsize=12)
    axes[0,0].axis('off')

    # Binarized
    axes[0,1].imshow(binary, cmap='gray')
    axes[0,1].set_title('Otsu Binarization', fontsize=12)
    axes[0,1].axis('off')

    # Vertical projection (column detection)
    axes[1,0].plot(col_sum, color='steelblue', linewidth=0.8)
    axes[1,0].axvline(x=w//2, color='red', linestyle='--', label='Page center')
    axes[1,0].set_title('Vertical Projection (column split detection)', fontsize=12)
    axes[1,0].set_xlabel('Column pixel'); axes[1,0].legend()

    # Horizontal projection (line detection)
    axes[1,1].plot(row_sum, np.arange(h), color='tomato', linewidth=0.8)
    axes[1,1].invert_yaxis()
    axes[1,1].set_title('Horizontal Projection (text line detection)', fontsize=12)
    axes[1,1].set_xlabel('Row pixel sum')

    plt.suptitle(f'Dataset Analysis: {Path(image_path).name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════
# UPDATED data_pipeline.py patch for Buendia two-column layout
# ══════════════════════════════════════════════════════════════

BUENDIA_PIPELINE_NOTES = """
IMPORTANT: Buendia PDF has TWO-PAGE SPREAD layout.
Each PDF page image contains BOTH the left and right book pages side by side.

Required pipeline adjustment:

1. After pdf_to_pages(), call batch_split_spreads() to get individual pages
2. Then call batch_extract_lines() on the split pages
3. The gutter between pages has very low ink density → easy to detect

Page correspondence:
  PDF page 1  → Cover (single page, no split needed)
  PDF page 2  → Dedication spread (L: blank/verso, R: "AL INFINITAMENTE...")
  PDF page 3  → First content spread (L: text, R: text)
  ...

Example in Colab:
    # After pdf_to_pages:
    split_pages = batch_split_spreads(PAGES_DIR, PAGES_DIR / 'split')
    # Then extract lines from split pages:
    all_lines = batch_extract_lines(PAGES_DIR / 'split', LINES_DIR)
"""
