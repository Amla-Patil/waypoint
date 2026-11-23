"""Turn a job description into weighted, individually checkable requirements."""
import re
from typing import Dict, List
from app.services.ner_service import extract_candidate_skills
from app.services import llm

DISPLAY = {"rest": "REST", "sql": "SQL", "aws": "AWS", "gcp": "GCP", "nlp": "NLP", "grpc": "gRPC", "oop": "OOP",
           "wms": "WMS", "erp": "ERP", "sap": "SAP", "ci/cd": "CI/CD", "bert": "BERT", "gpt": "GPT", "mlflow": "MLflow",
           "llms": "LLMs", "graphql": "GraphQL", "postgresql": "PostgreSQL", "mongodb": "MongoDB", "nodejs": "Node.js",
           "fastapi": "FastAPI", "xgboost": "XGBoost", "lightgbm": "LightGBM", "github actions": "GitHub Actions"}
def _display(n: str) -> str:
    return DISPLAY.get(n.lower(), n)

MUST = re.compile(r"\b(required|must|essential|mandatory|minimum|need(?:ed)?)\b", re.I)
NICE = re.compile(r"\b(preferred|nice to have|bonus|plus|desirable|good to have|optional)\b", re.I)

def extract_requirements(jd: str, use_llm: bool = False) -> List[Dict]:
    """Return [{name, weight, source}] with weights normalised to sum to 1.
    Dictionary skills found in the JD are weighted by the cue words on their line
    (must=2, neutral=1, nice-to-have=0.5)."""
    reqs: Dict[str, Dict] = {}
    lines = [l for l in re.split(r"[\n\r]+", jd) if l.strip()]
    for line in lines:
        found, _ = extract_candidate_skills(line)
        w = 2.0 if MUST.search(line) else 0.5 if NICE.search(line) else 1.0
        for name, _ctx in found:
            name = _display(name)
            if name not in reqs or w > reqs[name]["raw"]:
                reqs[name] = {"name": name, "raw": w, "source": "jd-dictionary"}
    if use_llm and llm.llm_available():
        data = llm.chat_json(
            "Extract the required skills from a job description. Respond with JSON only: "
            '{"requirements":[{"name":"<short skill name>","importance":1|2|3}]}. Max 12 items.',
            jd[:3000], 500)
        for r in (data or {}).get("requirements", []):
            n = str(r.get("name", "")).strip()
            if n and n not in reqs:
                reqs[n] = {"name": n, "raw": float(r.get("importance", 1)) / 2, "source": "llm"}
    low = jd.lower()
    items = sorted(reqs.values(), key=lambda r: (low.find(r["name"].lower()) if low.find(r["name"].lower()) >= 0 else 10**9, r["name"]))
    total = sum(r["raw"] for r in items) or 1.0
    return [{"name": r["name"], "weight": round(r["raw"] / total, 4), "source": r["source"]} for r in items]
