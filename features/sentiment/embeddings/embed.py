"""Text embedding models for quantitative sentiment analysis.

Provides:
- HashingTfidfEmbedder: deterministic, dependency-free default using scikit-learn
- TorchTextEmbedder: PyTorch/HuggingFace transformer wrapper with honest fallback
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


class HashingTfidfEmbedder:
    """Working default text embedder using scikit-learn HashingVectorizer + TfidfTransformer.

    Provides deterministic, fast text feature embeddings without external model weights.
    """

    def __init__(self, n_features: int = 64) -> None:
        """Initialize HashingTfidfEmbedder.

        Args:
            n_features: Number of output hash dimension features (default 64).
        """
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
        )
        self._transformer = TfidfTransformer()

    def fit(self, texts: Sequence[str]) -> HashingTfidfEmbedder:
        """Fit TF-IDF weighting on corpus of text documents."""
        counts = self._vectorizer.transform(texts)
        self._transformer.fit(counts)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform texts into normalized TF-IDF embedding vectors of shape (N, n_features)."""
        counts = self._vectorizer.transform(texts)
        # Handle case if transformer was not explicitly fitted beforehand
        try:
            tfidf = self._transformer.transform(counts)
        except Exception:
            self._transformer.fit(counts)
            tfidf = self._transformer.transform(counts)

        arr = tfidf.toarray()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        return arr / safe_norms

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """Fit and transform texts in a single call."""
        return self.fit(texts).transform(texts)


class TorchTextEmbedder:
    """PyTorch / HuggingFace text embedding loader with explicit dependency check.

    NOTE: This class dynamically attempts to load transformer weights (e.g. ProsusAI/finbert)
    ONLY if the third-party 'transformers' library is installed. If 'transformers' is missing,
    it raises an explicit, transparent ImportError directing users to HashingTfidfEmbedder.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        import importlib.util

        self.model_name = model_name
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            raise ImportError(
                "TorchTextEmbedder requires 'transformers' library which is not installed. "
                "Use HashingTfidfEmbedder as the supported, deterministic default embedder."
            )

        import importlib

        transformers = importlib.import_module("transformers")
        auto_model = getattr(transformers, "AutoModel")
        auto_tok = getattr(transformers, "AutoTokenizer")

        self.tokenizer = auto_tok.from_pretrained(model_name)
        self.model = auto_model.from_pretrained(model_name)
        self.model.eval()

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts using the underlying transformer."""
        import importlib

        torch = importlib.import_module("torch")

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings
