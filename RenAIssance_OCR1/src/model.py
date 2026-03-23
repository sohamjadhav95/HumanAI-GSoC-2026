"""
model.py
────────────────────────────────────────────────────────────────
RenAIssance OCR1 – GSoC 2026

Architecture:
  CNN Backbone (ResNet-18 feature extractor)
      ↓
  Adaptive pooling → sequence of column feature vectors
      ↓
  Bidirectional LSTM (2 layers)
      ↓
  Linear projection → per-timestep character logits
      ↓
  Weighted CTC Loss (training)  /  Constrained Beam Search (inference)

Key innovations for historical Spanish OCR:
  1. Weighted CTC loss  – upweights rare characters (ſ, diacritics, ligatures)
  2. Constrained beam search  – lexicon-guided decoding reduces hallucinations
────────────────────────────────────────────────────────────────
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. CNN BACKBONE
# ══════════════════════════════════════════════════════════════

class CNNBackbone(nn.Module):
    """
    ResNet-18 feature extractor adapted for variable-width line images.

    Input  : (B, 1, H, W)   H=64 fixed, W varies
    Output : (B, 512, H', W')  →  H' collapsed by adaptive pool

    Modifications vs. stock ResNet-18:
      - First conv: 1 input channel (grayscale)
      - Remove final FC layer and avgpool
      - Insert adaptive pool to collapse height to 1
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # ── Adapt first conv for grayscale ──────────────────
        first_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        # Initialise with mean of RGB weights so pretrained features transfer
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(
                    first_conv.weight.mean(dim=1, keepdim=True)
                )

        self.bn1    = backbone.bn1
        self.relu   = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Collapse height dimension → 1 (W dimension preserved for sequence)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # → (B, 64,  H/4,  W/4)
        x = self.layer1(x)        # → (B, 64,  H/4,  W/4)
        x = self.layer2(x)        # → (B, 128, H/8,  W/8)
        x = self.layer3(x)        # → (B, 256, H/16, W/16)
        x = self.layer4(x)        # → (B, 512, H/32, W/32)
        x = self.adaptive_pool(x) # → (B, 512, 1, W/32)
        x = x.squeeze(2)          # → (B, 512, W/32)  [channels, time]
        x = x.permute(2, 0, 1)    # → (W/32, B, 512)  [time, batch, feat]
        return x

    @staticmethod
    def output_length(input_width: int) -> int:
        """Calculate output sequence length for a given input width."""
        # ResNet-18 effective stride = 4 (maxpool) * 2^3 (3 stride-2 layers) = 32
        return max(1, input_width // 32)


# ══════════════════════════════════════════════════════════════
# 2. SEQUENCE MODELLING (BiLSTM)
# ══════════════════════════════════════════════════════════════

class BiLSTMEncoder(nn.Module):
    """
    Two-layer bidirectional LSTM.
    Input  : (T, B, input_size)
    Output : (T, B, hidden_size * 2)
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)   # (T, B, hidden*2)
        return out


# ══════════════════════════════════════════════════════════════
# 3. FULL CRNN
# ══════════════════════════════════════════════════════════════

class CRNN(nn.Module):
    """
    End-to-end CRNN for historical OCR.

      CNN  →  BiLSTM  →  Linear  →  log-softmax

    Trained with weighted CTC loss; decoded with constrained beam search.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        pretrained_cnn: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.cnn  = CNNBackbone(pretrained=pretrained_cnn)
        self.lstm = BiLSTMEncoder(512, hidden_size, lstm_layers, dropout)
        self.fc   = nn.Linear(hidden_size * 2, num_classes)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, H, W)  – batch of grayscale line images
        Returns:
            log_probs : (T, B, num_classes)  – log-softmax output for CTC
        """
        features = self.cnn(x)            # (T, B, 512)
        encoded  = self.lstm(features)    # (T, B, hidden*2)
        encoded  = self.drop(encoded)
        logits   = self.fc(encoded)       # (T, B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def input_lengths(self, pixel_widths: torch.Tensor) -> torch.Tensor:
        """Convert pixel widths to CNN output sequence lengths."""
        return torch.clamp(pixel_widths // 32, min=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════
# 4. WEIGHTED CTC LOSS
# ══════════════════════════════════════════════════════════════

class WeightedCTCLoss(nn.Module):
    """
    CTC loss with per-character class weights.

    Rare characters in 17th-century Spanish print (ſ, ã, ū, ligatures)
    receive higher loss weight so the model is penalised more for
    misrecognising them → forces the network to learn rare letterforms.

    Strategy:
      standard_ctc_loss  *  mean(weights[label_chars])
    where weights are inverse-frequency values (rare = high weight).

    Args:
        vocab          : {char: index} mapping
        char_weights   : {char: float} inverse-frequency weights
        blank_idx      : CTC blank token index (default 0)
        weight_scale   : global multiplier for the weighting effect
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        char_weights: Optional[Dict[str, float]] = None,
        blank_idx: int = 0,
        weight_scale: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.blank_idx    = blank_idx
        self.weight_scale = weight_scale
        self.ctc_loss     = nn.CTCLoss(
            blank=blank_idx, reduction="none", zero_infinity=True
        )

        # Build weight tensor indexed by vocab index
        num_classes = len(vocab)
        w = torch.ones(num_classes, dtype=torch.float32)
        if char_weights:
            for char, wt in char_weights.items():
                if char in vocab:
                    w[vocab[char]] = wt
        # Normalise so mean weight ≈ 1
        w = w / w.mean()
        self.register_buffer("class_weights", w)

    def forward(
        self,
        log_probs:      torch.Tensor,  # (T, B, C)
        targets:        torch.Tensor,  # (sum_label_lengths,)
        input_lengths:  torch.Tensor,  # (B,)
        target_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:

        # Per-sample CTC loss
        per_sample = self.ctc_loss(log_probs, targets,
                                   input_lengths, target_lengths)  # (B,)

        # Compute per-sample weight = mean of weights for each GT char
        sample_weights = []
        offset = 0
        for length in target_lengths:
            n = length.item()
            chars = targets[offset : offset + n]
            class_weights = self.class_weights.to(chars.device)
            w = class_weights[chars].mean()
            sample_weights.append(w)
            offset += n

        weights_tensor = torch.stack(sample_weights)           # (B,)
        weights_tensor = 1.0 + (weights_tensor - 1.0) * self.weight_scale

        weighted_loss = (per_sample * weights_tensor).mean()
        return weighted_loss


# ══════════════════════════════════════════════════════════════
# 5. CONSTRAINED BEAM SEARCH DECODER
# ══════════════════════════════════════════════════════════════

class ConstrainedBeamSearchDecoder:
    """
    CTC beam search with lexicon-based score penalty.

    At each step, each active beam hypothesis is scored against a
    lexicon trie.  Hypotheses that start to diverge from any word
    in the lexicon receive a configurable score penalty, reducing
    the probability of outputting hallucinated non-words.

    Args:
        vocab          : {char: index}
        beam_width     : number of beams to maintain
        lexicon_path   : path to Renaissance Spanish lexicon (.txt, one word per line)
        lm_weight      : relative weight of lexicon penalty vs. acoustic score
        blank_idx      : CTC blank token index
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        beam_width: int = 10,
        lexicon_path: Optional[str] = None,
        lm_weight: float = 0.5,
        blank_idx: int = 0,
    ):
        self.idx2char    = {v: k for k, v in vocab.items() if k != "<blank>"}
        self.blank_idx   = blank_idx
        self.beam_width  = beam_width
        self.lm_weight   = lm_weight
        self.lexicon_trie = self._build_trie(lexicon_path) if lexicon_path else None

    # ── Trie construction ──────────────────────────────────────
    def _build_trie(self, lexicon_path: str) -> Dict:
        """Build a character trie from the lexicon for O(k) prefix look-up."""
        trie = {}
        path = Path(lexicon_path)
        if not path.exists():
            logger.warning(f"Lexicon not found: {lexicon_path} – running unconstrained")
            return trie
        with open(lexicon_path, "r", encoding="utf-8") as f:
            for word in f:
                word = word.strip().lower()
                node = trie
                for char in word:
                    node = node.setdefault(char, {})
                node["__end__"] = True
        logger.info(f"Loaded lexicon trie from '{lexicon_path}'")
        return trie

    def _prefix_in_lexicon(self, prefix: str) -> Tuple[bool, bool]:
        """
        Returns (is_valid_prefix, is_complete_word).
        """
        if not self.lexicon_trie:
            return True, False
        node = self.lexicon_trie
        for char in prefix.lower():
            if char not in node:
                return False, False
            node = node[char]
        return True, "__end__" in node

    # ── Standard CTC beam search (no lexicon) ─────────────────
    def _ctc_beam_search(
        self, probs: torch.Tensor
    ) -> str:
        """
        Standard CTC beam search without lexicon.
        probs: (T, num_classes) — softmax probabilities for one sample
        """
        T, C = probs.shape
        # beam: {text: (prob_blank, prob_no_blank)}
        beams = {("",): (1.0, 0.0)}

        for t in range(T):
            new_beams: Dict = {}
            for text, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = probs[t, c].item()
                    if c == self.blank_idx:
                        new_text = text
                        new_p_b  = (p_b + p_nb) * p
                        new_p_nb = new_beams.get(new_text, (0.0, 0.0))[1]
                        new_beams[new_text] = (
                            new_beams.get(new_text, (0.0, 0.0))[0] + new_p_b,
                            new_p_nb,
                        )
                    else:
                        char = self.idx2char.get(c, "")
                        new_text = text + (char,)
                        if text and text[-1] == c:
                            new_p_nb = p_nb * p
                        else:
                            new_p_nb = (p_b + p_nb) * p
                        existing = new_beams.get(new_text, (0.0, 0.0))
                        new_beams[new_text] = (existing[0], existing[1] + new_p_nb)

            # Keep top-K beams
            beams = dict(
                sorted(new_beams.items(),
                       key=lambda x: x[1][0] + x[1][1],
                       reverse=True)[: self.beam_width]
            )

        best = max(beams, key=lambda t: beams[t][0] + beams[t][1])
        return "".join(best)

    # ── Constrained beam search ────────────────────────────────
    def decode(
        self, log_probs: torch.Tensor, input_length: int
    ) -> str:
        """
        Decode a single sample with constrained beam search.

        Args:
            log_probs    : (T, num_classes) — log-softmax output
            input_length : valid length (T may be padded)

        Returns:
            Decoded string
        """
        probs = torch.exp(log_probs[:input_length])  # (T, C)

        if self.lexicon_trie is None:
            return self._ctc_beam_search(probs)

        # ── Lexicon-constrained beam ─────────────────────────
        # beam state: (text_so_far, current_word_prefix, p_blank, p_no_blank)
        beams = [("", "", 1.0, 0.0)]
        T, C = probs.shape

        for t in range(T):
            candidates: Dict = {}
            for text, word_pfx, p_b, p_nb in beams:
                for c in range(C):
                    p = probs[t, c].item()
                    if c == self.blank_idx:
                        key = (text, word_pfx)
                        old = candidates.get(key, (0.0, 0.0))
                        candidates[key] = (old[0] + (p_b + p_nb) * p, old[1])
                    else:
                        char = self.idx2char.get(c, "")
                        if char == " ":
                            # Word boundary: validate completed word
                            is_pfx, is_word = self._prefix_in_lexicon(word_pfx)
                            lex_penalty = 0.0 if is_word else self.lm_weight
                            new_text  = text + " "
                            new_pfx   = ""
                        else:
                            new_pfx   = word_pfx + char
                            is_pfx, _ = self._prefix_in_lexicon(new_pfx)
                            # Penalise if prefix is not in lexicon
                            lex_penalty = 0.0 if is_pfx else self.lm_weight
                            new_text  = text + char

                        # Emission prob discounted by lexicon penalty
                        emission_p = p * math.exp(-lex_penalty)

                        # CTC transition
                        if text and text[-1] == char:
                            new_p_nb = p_nb * emission_p
                        else:
                            new_p_nb = (p_b + p_nb) * emission_p

                        key = (new_text, new_pfx if char != " " else "")
                        old = candidates.get(key, (0.0, 0.0))
                        candidates[key] = (old[0], old[1] + new_p_nb)

            # Prune to beam_width
            beams = sorted(
                [(t, wp, pb, pnb) for (t, wp), (pb, pnb) in candidates.items()],
                key=lambda x: x[2] + x[3],
                reverse=True,
            )[: self.beam_width]

        if not beams:
            return ""
        best = max(beams, key=lambda x: x[2] + x[3])
        return best[0].strip()

    def decode_batch(
        self,
        log_probs: torch.Tensor,      # (T, B, C)
        input_lengths: torch.Tensor,  # (B,)
    ) -> List[str]:
        """Decode a full batch, one sample at a time."""
        B = log_probs.shape[1]
        return [
            self.decode(log_probs[:, b, :], input_lengths[b].item())
            for b in range(B)
        ]


# ══════════════════════════════════════════════════════════════
# 6. GREEDY DECODER (fast inference / debugging)
# ══════════════════════════════════════════════════════════════

def greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    idx2char: Dict[int, str],
    blank_idx: int = 0,
) -> List[str]:
    """
    Argmax / greedy CTC decoding.
    Fast alternative to beam search; used during training for monitoring.

    Args:
        log_probs     : (T, B, C)
        input_lengths : (B,)  – valid T lengths
        idx2char      : {index: char}
        blank_idx     : blank token index

    Returns:
        List of decoded strings, length B
    """
    # (T, B) argmax indices
    indices = log_probs.argmax(dim=-1)
    B = indices.shape[1]
    results = []

    for b in range(B):
        length = input_lengths[b].item()
        seq = indices[:length, b].tolist()

        # CTC collapse: remove blanks and repeated chars
        decoded = []
        prev = None
        for idx in seq:
            if idx != blank_idx and idx != prev:
                char = idx2char.get(idx, "")
                if char:
                    decoded.append(char)
            prev = idx
        results.append("".join(decoded))

    return results


# ══════════════════════════════════════════════════════════════
# 7. MODEL FACTORY & CHECKPOINTING
# ══════════════════════════════════════════════════════════════

def build_model(
    vocab: Dict[str, int],
    hidden_size: int = 256,
    lstm_layers: int = 2,
    dropout: float = 0.3,
    pretrained_cnn: bool = True,
) -> CRNN:
    """Construct CRNN and log parameter count."""
    num_classes = len(vocab)
    model = CRNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        pretrained_cnn=pretrained_cnn,
    )
    logger.info(f"CRNN built | classes={num_classes} | "
                f"params={model.count_parameters():,}")
    return model


def save_checkpoint(
    model: CRNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_cer: float,
    path: str,
) -> None:
    torch.save(
        {
            "epoch":     epoch,
            "val_cer":   val_cer,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint saved  →  {path}  (epoch {epoch}, CER {val_cer:.4f})")


def load_checkpoint(
    path: str,
    model: CRNN,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    logger.info(f"Loaded checkpoint '{path}' (epoch {ckpt['epoch']}, "
                f"CER {ckpt['val_cer']:.4f})")
    return ckpt["epoch"], ckpt["val_cer"]
