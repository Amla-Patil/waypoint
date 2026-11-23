"""Module 2: retrieval over authoritative knowledge (RAG retrieval half).

Corpus (three kinds, each record keeps its source id so results are citable):
  occupation : O*NET occupations (title, description, top skills)   -> grounding for the target role
  module     : the 28-module course catalogue (with prerequisites)    -> learning units
  course     : Coursera dataset (3,522 courses)                        -> concrete resources
Retrieval is hybrid: dense similarity (embedding_service) fused with word-level TF-IDF by
reciprocal-rank fusion. Embeddings are cached on disk per backend.
"""
import hashlib, json, os
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.embedding_service import get_embedder, dense_dot
from app.services import course_search

_HERE = os.path.dirname(__file__)
_APP = os.path.join(_HERE, "..")
_CACHE_DIR = os.path.join(_APP, ".cache")

_docs: List[Dict] = []
_dense: Optional[np.ndarray] = None
_word: Optional[TfidfVectorizer] = None
_word_m = None
_built = False

def _build_corpus() -> List[Dict]:
    docs: List[Dict] = []
    try:
        for o in json.load(open(os.path.join(_APP, "onet_occupations.json"), encoding="utf-8")):
            docs.append({"kind": "occupation", "id": o["code"], "title": o["title"],
                         "text": f"{o['title']}. {o['description']} Key skills: {', '.join(o['top_skills'])}",
                         "meta": {"top_skills": o["top_skills"]}})
    except FileNotFoundError:
        print("[RAG] onet_occupations.json missing - run ml-engine/export_onet_for_rag.py")
    for m in json.load(open(os.path.join(_APP, "course_catalog.json"), encoding="utf-8")):
        docs.append({"kind": "module", "id": m["id"], "title": m["title"],
                     "text": f"{m['title']}. {m['description']} Tags: {', '.join(m['tags'])}",
                     "meta": {"hours": m["hours"], "prerequisites": m["prerequisites"]}})
    course_search._load()
    for i, c in enumerate(course_search._COURSES):
        docs.append({"kind": "course", "id": f"coursera:{i}", "title": c["title"],
                     "text": f"{c['title']}. Skills: {', '.join(c['skills'][:12])}. {c['desc_lower'][:200]}",
                     "meta": {"url": c["url"], "university": c["university"], "rating": c["rating"],
                              "difficulty": c["difficulty"]}})
    return docs

def _ensure_built():
    global _docs, _dense, _word, _word_m, _built
    if _built:
        return
    emb = get_embedder()
    _docs = _build_corpus()
    texts = [d["text"] for d in _docs]
    if emb.name == "tfidf":
        emb.fit(texts)
    sig = hashlib.md5((emb.name + os.environ.get("EMBEDDING_MODEL", "") + str(len(texts)) + texts[0][:50] + texts[-1][:50]).encode()).hexdigest()[:12]
    cache = os.path.join(_CACHE_DIR, f"rag_{emb.name}_{sig}.npy")
    if emb.name == "sbert" and os.path.exists(cache):
        _dense = np.load(cache)
    else:
        _dense = emb.encode(texts)
        if emb.name == "sbert":
            os.makedirs(_CACHE_DIR, exist_ok=True)
            np.save(cache, _dense)
    _word = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    _word_m = _word.fit_transform(texts)
    _built = True
    print(f"[RAG] indexed {len(_docs)} records with backend={emb.name}")

def index_size() -> Dict[str, int]:
    _ensure_built()
    out: Dict[str, int] = {}
    for d in _docs:
        out[d["kind"]] = out.get(d["kind"], 0) + 1
    return out

def retrieve(query: str, k: int = 5, kinds: Optional[List[str]] = None, min_dense: float = 0.0) -> List[Dict]:
    """Hybrid retrieval. Each hit: kind, id, title, score (dense cosine), fused rank score, meta."""
    _ensure_built()
    emb = get_embedder()
    q = emb.encode([query])
    dense = dense_dot(_dense, q).ravel()
    lex = (_word_m @ _word.transform([query]).T).toarray().ravel()
    mask = np.array([(kinds is None or d["kind"] in kinds) for d in _docs])
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    r_dense = {i: r for r, i in enumerate(idx[np.argsort(-dense[idx])])}
    r_lex = {i: r for r, i in enumerate(idx[np.argsort(-lex[idx])])}
    fused = {i: 1.0 / (60 + r_dense[i]) + 1.0 / (60 + r_lex[i]) for i in idx}
    out = []
    for i in sorted(fused, key=fused.get, reverse=True):
        if dense[i] < min_dense and lex[i] <= 0:
            continue
        d = _docs[i]
        out.append({"kind": d["kind"], "id": d["id"], "title": d["title"], "similarity": round(float(dense[i]), 3),
                    "lexical": round(float(lex[i]), 3), "meta": d["meta"]})
        if len(out) == k:
            break
    return out
