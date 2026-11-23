from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import extract, catalog, courses, match, rag
from app.services import llm
from app.services.embedding_service import get_embedder

app = FastAPI(
    title="WayPoint API",
    description="Explainable, RAG-grounded resume-job matching with skill-gap recommendation",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(extract.router, prefix="/extract", tags=["Resume parsing"])
app.include_router(match.router,   prefix="/match",   tags=["Matching"])
app.include_router(rag.router,     prefix="/rag",     tags=["Retrieval"])
app.include_router(catalog.router, prefix="/catalog", tags=["Catalogue"])
app.include_router(courses.router, prefix="/courses", tags=["Courses"])

@app.get("/health")
def health():
    return {"status": "ok", "app": "WayPoint", "embedding_backend": get_embedder().name, "llm_available": llm.llm_available()}
