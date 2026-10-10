"""Financial sentiment lexicon scoring with negation handling.

Includes a transparent DEMO_LEXICON subset (~40 positive, ~40 negative, ~10 negators)
for testing and offline scoring, plus a loader for the real Loughran-McDonald master dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence
import pandas as pd

# Transparent DEMO subset clearly labeled as NOT full Loughran-McDonald
DEMO_LEXICON: dict[str, set[str]] = {
    "positive": {
        "profit", "profitable", "growth", "strong", "gain", "gains", "surged", "surge",
        "record", "beat", "exceeded", "higher", "positive", "outperformed", "dividend",
        "opportunity", "robust", "resilient", "success", "successful", "upside", "expansion",
        "improving", "improved", "innovation", "innovative", "bullish", "rally", "confidence",
        "momentum", "recovery", "advantage", "favorable", "optimistic", "breakthrough",
        "milestone", "solid", "thriving", "lucrative", "strength",
    },
    "negative": {
        "loss", "losses", "decline", "declined", "weak", "weakness", "drop", "dropped",
        "fell", "slump", "miss", "missed", "lower", "negative", "underperformed", "deficit",
        "risk", "risks", "default", "bankruptcy", "downgrade", "downgraded", "bearish",
        "crash", "recession", "inflation", "headwinds", "litigation", "lawsuit", "penalty",
        "fraud", "investigation", "deteriorating", "uncertainty", "vulnerable", "cut",
        "restructuring", "layoff", "layoffs", "plunge", "plunged",
    },
    "negators": {
        "not", "never", "no", "neither", "hardly", "scarcely", "barely", "cannot", "without", "none",
    },
}


class LexiconScorer:
    """Financial text sentiment analyzer using word polarity and contextual negator flipping."""

    def __init__(
        self,
        positive: set[str] | Sequence[str] | None = None,
        negative: set[str] | Sequence[str] | None = None,
        negators: set[str] | Sequence[str] | None = None,
    ) -> None:
        """Initialize LexiconScorer.

        Defaults to the transparent DEMO_LEXICON subset if no custom lists are provided.
        """
        self.positive = set(positive) if positive is not None else DEMO_LEXICON["positive"]
        self.negative = set(negative) if negative is not None else DEMO_LEXICON["negative"]
        self.negators = set(negators) if negators is not None else DEMO_LEXICON["negators"]

    def score(self, text: str) -> float:
        """Compute polarized sentiment score for a text string in [-1.0, 1.0].

        Formula:
            score = (pos_count - neg_count) / (pos_count + neg_count + 1e-9)

        If a negator appears within 3 preceding tokens, the sentiment polarity of
        the target word is flipped (positive becomes negative, negative becomes positive).

        Args:
            text: Input document or financial headline.

        Returns:
            float: Normalized sentiment score in [-1.0, 1.0].
        """
        if not text or not isinstance(text, str):
            return 0.0

        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not tokens:
            return 0.0

        pos_count = 0.0
        neg_count = 0.0

        for i, token in enumerate(tokens):
            is_pos = token in self.positive
            is_neg = token in self.negative

            if not is_pos and not is_neg:
                continue

            # Look up to 3 preceding tokens for negators
            window_start = max(0, i - 3)
            preceding_tokens = tokens[window_start:i]
            has_negator = any(neg in self.negators for neg in preceding_tokens)

            if has_negator:
                # Polarity flipped
                if is_pos:
                    neg_count += 1.0
                elif is_neg:
                    pos_count += 1.0
            else:
                if is_pos:
                    pos_count += 1.0
                elif is_neg:
                    neg_count += 1.0

        tot = pos_count + neg_count
        if tot == 0.0:
            return 0.0

        score_val = (pos_count - neg_count) / (tot + 1e-9)
        return max(-1.0, min(1.0, float(score_val)))


def load_lm_csv(path: str | Path) -> LexiconScorer:
    """Load official Loughran-McDonald Master Dictionary CSV if supplied by the user.

    Expects columns: 'Word', 'Positive', 'Negative' (non-zero indicates category).

    Args:
        path: Path to Loughran-McDonald CSV file.

    Returns:
        LexiconScorer configured with full Loughran-McDonald dictionary.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Loughran-McDonald dictionary not found at {path}")

    df = pd.read_csv(p)
    word_col = "Word" if "Word" in df.columns else df.columns[0]

    pos_words = set(df[df["Positive"] > 0][word_col].astype(str).str.lower())
    neg_words = set(df[df["Negative"] > 0][word_col].astype(str).str.lower())

    return LexiconScorer(
        positive=pos_words,
        negative=neg_words,
        negators=DEMO_LEXICON["negators"],
    )
