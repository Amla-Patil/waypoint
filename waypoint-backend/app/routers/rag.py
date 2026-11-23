from typing import List, Optional
from fastapi import APIRouter, Query
from app.services import rag_service
from app.services.embedding_service import get_embedder

router = APIRouter()

@router.get("/search", summary="Hybrid retrieval over O*NET occupations, catalogue modules and Coursera courses")
def search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20),
           kind: Optional[str] = Query(None, description="occupation | module | course")):
    return {"query": q, "backend": get_embedder().name, "hits": rag_service.retrieve(q, k=k, kinds=[kind] if kind else None)}

@router.get("/stats", summary="Index size per source")
def stats():
    return {"backend": get_embedder().name, "records": rag_service.index_size()}
