import re
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.models import MatchRequest, MatchResponse
from app.services import explain_service as ex, fairness_service, graph_service, rag_service
from app.services.embedding_service import get_embedder
from app.services.pdf_service import extract_text_from_upload
from app.services.requirements_service import extract_requirements

router = APIRouter()

def _role_title(jd: str) -> str:
    for line in [l.strip() for l in jd.split("\n") if l.strip()][:5]:
        m = re.match(r"(?:role|position|title)\s*:\s*(.+)", line, re.I)
        if m:
            return m.group(1).strip()[:80]
        if len(line) < 60 and "." not in line and "," not in line and ":" not in line:
            return line
    return "Target role"

def run_match(resume: str, jd: str, use_llm: bool = False) -> dict:
    if not resume.strip() or not jd.strip():
        raise HTTPException(400, "Both resume text and job description are required.")
    reqs = extract_requirements(jd, use_llm=use_llm)
    if not reqs:
        raise HTTPException(422, "No recognisable skill requirements found in the job description. "
                                 "List the required skills (e.g. 'Required: Python, SQL').")
    emb = get_embedder()
    results, mode = ex.match_requirements(resume, reqs, use_llm=use_llm)
    score = ex.score_results(results)
    title = _role_title(jd)
    gaps = [r["requirement"] for r in results if r["verdict"] == "not_met"]
    partial = [r["requirement"] for r in results if r["verdict"] == "partial"]
    met = [r["requirement"] for r in results if r["verdict"] == "met"]
    grounding = [{"code": h["id"], "title": h["title"], "similarity": h["similarity"],
                  "top_skills": h["meta"]["top_skills"]} for h in rag_service.retrieve(title + " " + jd[:300], k=3, kinds=["occupation"])]
    path = graph_service.build_pathway(gaps, partial, met)
    trace = ex.build_trace(results, score, mode, emb.name) + [
        f"[RAG] retrieved {len(grounding)} O*NET occupations for grounding: " + "; ".join(g["title"] for g in grounding)] + path["trace"]
    return {"role_title": title, "match_score": score, "baseline_keyword_score": ex.keyword_baseline(resume, reqs),
            "semantic_similarity": ex.document_similarity(resume, jd), "embedding_backend": emb.name, "mode": mode,
            "requirements": results, "skill_gaps": gaps, "partial_skills": partial, "met_skills": met, "grounding": grounding,
            "modules": path["modules"], "resources": path["resources"], "total_hours": path["total_hours"],
            "full_hours": path["full_hours"], "hours_saved": path["hours_saved"], "reasoning_trace": trace}

@router.post("/text", response_model=MatchResponse, summary="Explainable resume-job match (text input)")
def match_text(req: MatchRequest):
    return run_match(req.resume_text, req.job_description, req.use_llm)

@router.post("", response_model=MatchResponse, summary="Explainable resume-job match (resume file upload)")
async def match_file(file: UploadFile = File(...), job_description: str = Form(...), use_llm: bool = Form(False)):
    return run_match(await extract_text_from_upload(file), job_description, use_llm)

@router.post("/fairness", summary="Counterfactual fairness and robustness check of the match score")
def fairness(req: MatchRequest):
    reqs = extract_requirements(req.job_description, use_llm=req.use_llm)
    if not reqs:
        raise HTTPException(422, "No recognisable skill requirements found in the job description.")
    out = fairness_service.run_checks(req.resume_text, reqs)
    out["embedding_backend"] = get_embedder().name
    return out
