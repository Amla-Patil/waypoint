"""Text embeddings behind one interface (Module 1: semantic matching).

Backends
  sbert : sentence-transformers (default model all-MiniLM-L6-v2) - the intended backend.
  tfidf : character n-gram TF-IDF - dependency-light fallback used when the model cannot be
          loaded (e.g. offline). It is lexical-semantic only, so thresholds differ per backend.
Select with EMBEDDING_BACKEND=auto|sbert|tfidf and EMBEDDING_MODEL=<hf model name>.
"""
import os
from typing import List, Optional
import numpy as np

# Calibration defaults (verdict thresholds on cosine similarity). These are starting points:
# calibrate on labelled skill pairs (see report, Chapter 3) before quoting accuracy.
THRESHOLDS = {
    "sbert": {"met": 0.55, "partial": 0.40},
    "tfidf": {"met": 0.60, "partial": 0.45},   # char n-gram overlap is not semantics: effectively lexical-only
}

def dense_dot(A, B) -> np.ndarray:
    """A @ B.T as a dense ndarray for dense or sparse inputs."""
    M = A @ B.T
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)

class Embedder:
    def __init__(self, backend: Optional[str] = None):
        want = (backend or os.environ.get("EMBEDDING_BACKEND", "auto")).lower()
        self.name = "tfidf"
        self._model = None
        self._tfidf = None
        if want in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                self._model = SentenceTransformer(model)
                self.name = "sbert"
            except Exception as e:
                if want == "sbert":
                    raise
                print(f"[EMBED] sentence-transformers unavailable ({type(e).__name__}); using TF-IDF fallback")
        self.thresholds = THRESHOLDS[self.name]

    # -- encoding ------------------------------------------------------------------------
    def fit(self, corpus: List[str]):
        """Fit the vocabulary (TF-IDF only). No-op for SBERT."""
        if self.name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
            self._tfidf.fit(corpus)
        return self

    def encode(self, texts: List[str]):
        """ndarray (sbert) or scipy sparse matrix (tfidf); both support `@` and `dense_dot`."""
        if not texts:
            return np.zeros((0, 1))
        if self.name == "sbert":
            return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        if self._tfidf is None:
            self.fit(texts)
        return self._tfidf.transform(texts)      # sparse, rows already L2-normalised by TfidfVectorizer

    def similarity(self, a: List[str], b: List[str]) -> np.ndarray:
        """Cosine similarity matrix (len(a) x len(b)). For TF-IDF the vocabulary is fitted on a+b."""
        if self.name == "tfidf":      # local vocabulary: must not disturb the fitted RAG index vectoriser
            from sklearn.feature_extraction.text import TfidfVectorizer
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(list(a) + list(b))
            return dense_dot(v.transform(a), v.transform(b))
        return dense_dot(self.encode(a), self.encode(b))

_default: Optional[Embedder] = None
def get_embedder() -> Embedder:
    global _default
    if _default is None:
        _default = Embedder()
    return _default
