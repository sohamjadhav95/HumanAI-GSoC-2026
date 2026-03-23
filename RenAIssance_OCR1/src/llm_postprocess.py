"""
llm_postprocess.py
────────────────────────────────────────────────────────────────
RenAIssance OCR1 – GSoC 2026

Late-stage LLM correction using Gemini 1.5 Pro (or GPT-4 fallback).

As specified in the OCR1 proposal, the LLM is used AFTER the
CRNN pipeline, not during it (contrast with Test II).

Pipeline position:
  CRNN raw output
       ↓
  Beam-search decoded text
       ↓
  [THIS MODULE] Gemini / GPT-4 contextual correction
       ↓
  Final transcription

Correction strategy:
  The LLM receives the raw OCR output and is prompted with
  domain-specific context about 17th-century Spanish printed text,
  enabling it to:
    1. Resolve ambiguous characters (e.g. ı → i, long-s ſ → s)
    2. Complete partially recognised words using language context
    3. Normalise archaic spellings to consistent historical forms
    4. Flag low-confidence regions for human review
────────────────────────────────────────────────────────────────
"""

import os
import time
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert in 17th-century early modern Spanish printed texts.
You are assisting in correcting the output of an OCR system that has read scanned pages
from historical Spanish printed sources.

Your task:
Given raw OCR output, correct recognition errors while following these strict rules:

RULES:
1. PRESERVE archaic spellings (e.g., "vn", "q̃", "dēllos", "señor" with tilde-n).
   Do NOT modernise spelling unless it is clearly a misrecognition.
2. PRESERVE original punctuation and capitalisation where sensible.
3. CORRECT obvious OCR errors such as:
   - Confused characters: rn → m,  cl → d,  li → h,  ii → n
   - Long-s confusion: ſ may have been read as f or s
   - Broken letters: partial characters that were split by the OCR
   - Garbled sequences from degraded ink or paper damage
4. DO NOT add, remove, or paraphrase content — only fix recognition errors.
5. DO NOT add punctuation that was not in the original.
6. If a word is completely unrecognisable, output [???] as a placeholder.
7. Return ONLY the corrected text. No explanation, no preamble.

CONTEXT:
- Language: Early Modern Spanish (17th century)
- Script type: Printed (not handwritten)
- Common issues: long-s (ſ), abbreviation tildes, degraded ink
- Source type: prose, historical/literary"""


LINE_PROMPT_TEMPLATE = """Correct the following OCR output from a 17th-century Spanish printed source:

OCR OUTPUT:
{ocr_text}

CORRECTED TEXT:"""


BATCH_PROMPT_TEMPLATE = """Correct each line of OCR output from a 17th-century Spanish printed source.
Return ONLY the corrected lines, one per line, in the same order.

OCR OUTPUT LINES:
{ocr_lines}

CORRECTED LINES:"""


# ══════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ══════════════════════════════════════════════════════════════

class GeminiCorrector:
    """
    Uses Gemini 1.5 Pro to post-correct OCR output.

    Args:
        api_key   : Google AI API key (or set GOOGLE_API_KEY env var)
        model     : Gemini model name
        batch_size: number of lines to correct in one API call
        max_retries: retry on rate-limit or transient errors
    """

    MODEL = "gemini-1.5-pro"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        batch_size: int = 20,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model       = model
        self.batch_size  = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            logger.warning(
                "No GOOGLE_API_KEY found. Set it via:\n"
                "  import os; os.environ['GOOGLE_API_KEY'] = 'YOUR_KEY'\n"
                "  or pass api_key= to GeminiCorrector()"
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(
                model_name=model,
                system_instruction=SYSTEM_PROMPT,
            )
            logger.info(f"Gemini client initialised ({model})")
        except ImportError:
            raise ImportError(
                "Install google-generativeai:  pip install google-generativeai"
            )

    def correct_line(self, ocr_text: str) -> str:
        """Correct a single OCR line. Returns corrected string."""
        prompt = LINE_PROMPT_TEMPLATE.format(ocr_text=ocr_text.strip())
        for attempt in range(self.max_retries):
            try:
                response = self._client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Gemini retry {attempt+1}: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Gemini failed after {self.max_retries} retries: {e}")
                    return ocr_text  # fall back to raw OCR

    def correct_batch(self, ocr_lines: List[str]) -> List[str]:
        """
        Correct a batch of OCR lines in one API call (more efficient).
        Falls back to line-by-line if batch parsing fails.
        """
        if not ocr_lines:
            return []

        lines_text = "\n".join(
            f"{i+1}. {line}" for i, line in enumerate(ocr_lines)
        )
        prompt = BATCH_PROMPT_TEMPLATE.format(ocr_lines=lines_text)

        for attempt in range(self.max_retries):
            try:
                response = self._client.generate_content(prompt)
                corrected = self._parse_batch_response(
                    response.text, len(ocr_lines)
                )
                if len(corrected) == len(ocr_lines):
                    return corrected
                else:
                    logger.warning(
                        f"Batch parse mismatch ({len(corrected)} vs "
                        f"{len(ocr_lines)}), falling back to line-by-line"
                    )
                    break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Gemini batch retry {attempt+1}: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Gemini batch failed: {e}")
                    break

        # Line-by-line fallback
        return [self.correct_line(line) for line in ocr_lines]

    @staticmethod
    def _parse_batch_response(text: str, expected: int) -> List[str]:
        """
        Parse numbered batch response.
        Accepts formats: "1. text", "1) text", or bare lines.
        """
        # Try numbered format first
        numbered = re.findall(r"^\d+[.)]\s*(.+)$", text, re.MULTILINE)
        if len(numbered) == expected:
            return numbered
        # Fall back to splitting by newline
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines

    def correct_document(
        self,
        ocr_lines: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Correct all OCR lines of a document, batched for API efficiency.

        Args:
            ocr_lines     : list of raw OCR output strings (one per line)
            show_progress : show tqdm progress bar

        Returns:
            list of corrected strings (same length as input)
        """
        from tqdm import tqdm

        corrected = []
        batches = [
            ocr_lines[i : i + self.batch_size]
            for i in range(0, len(ocr_lines), self.batch_size)
        ]

        iterator = tqdm(batches, desc="LLM correction") if show_progress else batches
        for batch in iterator:
            corrected.extend(self.correct_batch(batch))

        logger.info(
            f"LLM post-processing complete: {len(corrected)} lines corrected"
        )
        return corrected


# ══════════════════════════════════════════════════════════════
# GPT-4 FALLBACK
# ══════════════════════════════════════════════════════════════

class GPT4Corrector:
    """
    OpenAI GPT-4 fallback corrector. Same interface as GeminiCorrector.
    """

    MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        batch_size: int = 20,
    ):
        self.batch_size = batch_size
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self.model   = model
            logger.info(f"GPT-4 client initialised ({model})")
        except ImportError:
            raise ImportError("Install openai:  pip install openai")

    def correct_line(self, ocr_text: str) -> str:
        prompt = LINE_PROMPT_TEMPLATE.format(ocr_text=ocr_text.strip())
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4 error: {e}")
            return ocr_text

    def correct_document(
        self, ocr_lines: List[str], show_progress: bool = True
    ) -> List[str]:
        from tqdm import tqdm
        iterator = tqdm(ocr_lines, desc="GPT-4 correction") if show_progress else ocr_lines
        return [self.correct_line(line) for line in iterator]


# ══════════════════════════════════════════════════════════════
# CORRECTION DELTA ANALYSIS
# ══════════════════════════════════════════════════════════════

def correction_delta_report(
    raw_ocr:   List[str],
    corrected: List[str],
    targets:   List[str],
) -> None:
    """
    Print a report showing how much LLM correction improved CER/WER.
    This is the core ablation result for the LLM post-processing stage.
    """
    from evaluate import compute_cer, compute_wer

    cer_before = compute_cer(raw_ocr,   targets)
    cer_after  = compute_cer(corrected, targets)
    wer_before = compute_wer(raw_ocr,   targets)
    wer_after  = compute_wer(corrected, targets)

    print("\n" + "=" * 55)
    print("LLM POST-PROCESSING DELTA REPORT")
    print("=" * 55)
    print(f"  {'Metric':<8}  {'Before':>8}  {'After':>8}  {'Δ':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'CER':<8}  {cer_before:>8.4f}  {cer_after:>8.4f}  "
          f"{cer_after - cer_before:>+8.4f}")
    print(f"  {'WER':<8}  {wer_before:>8.4f}  {wer_after:>8.4f}  "
          f"{wer_after - wer_before:>+8.4f}")
    print("=" * 55)

    # Show 5 examples where LLM helped most
    from evaluate import per_sample_cer, _edit_distance
    raw_cers  = per_sample_cer(raw_ocr,   targets)
    post_cers = per_sample_cer(corrected, targets)
    gains = [(r - p, i) for i, (r, p) in enumerate(zip(raw_cers, post_cers))]
    gains.sort(reverse=True)

    print("\nTop-5 most improved lines:")
    for gain, idx in gains[:5]:
        print(f"  Gain +{gain:.3f}")
        print(f"    GT  : {targets[idx]}")
        print(f"    RAW : {raw_ocr[idx]}")
        print(f"    LLM : {corrected[idx]}")
        print()
