"""Offline tests (no API key, no model download). Run: pytest -q  (from waypoint-backend/)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from app.main import app
from app.services import graph_service
from app.services.explain_service import verify_evidence, lexical_hit

client = TestClient(app)
JD = """ML Engineer
Required: Python, PyTorch, REST APIs
Must have experience with Docker and Kubernetes
Nice to have: MLflow, Spark"""
RESUME = """Priya Sharma
- Built RESTful services in Flask for model serving
- 3 years Python; trained models in PyTorch
- Basic Docker usage in a college project
- Led a team of five"""

def match(resume=RESUME, jd=JD):
    r = client.post("/match/text", json={"resume_text": resume, "job_description": jd}); assert r.status_code == 200, r.text
    return r.json()

def test_health_and_removed_features():
    assert client.get("/health").json()["status"] == "ok"
    for path in ("/score", "/skill-test/adaptive", "/pathway/text"):
        assert client.post(path, json={}).status_code == 404

def test_every_evidence_quote_is_verified():
    d = match()
    for r in d["requirements"]:
        if r["evidence"]:
            assert r["evidence_verified"] is True and verify_evidence(RESUME, r["evidence"])

def test_score_is_deterministic_and_computed_in_code():
    a, b = match(), match()
    assert a["match_score"] == b["match_score"]
    tw = sum(r["weight"] for r in a["requirements"])
    assert abs(a["match_score"] - round(100 * sum(r["weight"] * r["score"] for r in a["requirements"]) / tw, 1)) < 0.11

def test_expected_verdicts():
    v = {r["requirement"]: r["verdict"] for r in match()["requirements"]}
    assert v["PyTorch"] == "met" and v["REST"] == "met"          # 'RESTful' matches 'REST'
    assert v["Docker"] == "partial"                               # 'Basic Docker' = learner wording
    assert v["Kubernetes"] == "not_met"

def test_fabricated_evidence_is_rejected():
    assert not verify_evidence(RESUME, "Expert in Kubernetes operators")
    assert lexical_hit("REST APIs", "Built RESTful services") and not lexical_hit("Kubernetes", "Used Docker")

def test_pathway_prerequisites_precede_dependents_and_no_invented_modules():
    d = match()
    pos = {m["module_id"]: i for i, m in enumerate(d["modules"])}
    for m in d["modules"]:
        assert m["module_id"] in graph_service.CATALOG_BY_ID                   # only catalogue modules
        assert all(pos[p] < pos[m["module_id"]] for p in m["prerequisites"])    # topological order
    assert d["total_hours"] <= d["full_hours"] and d["hours_saved"] == d["full_hours"] - d["total_hours"]

def test_unverifiable_course_links_are_not_invented():
    for m in match()["modules"]:
        assert (m["course_url"] is None) or m["course_url"].startswith("http")

def test_empty_inputs_and_unparseable_jd():
    assert client.post("/match/text", json={"resume_text": "", "job_description": JD}).status_code == 400
    assert client.post("/match/text", json={"resume_text": RESUME, "job_description": "We are a friendly team."}).status_code == 422

def test_fairness_endpoint():
    r = client.post("/match/fairness", json={"resume_text": RESUME, "job_description": JD}); assert r.status_code == 200
    d = r.json()
    assert len(d["identity"]) == 4 and d["identity_pass"] is True
    assert {x["variant"] for x in d["robustness"]} >= {"keyword stuffing appended"}

def test_rag_retrieval_returns_citable_sources():
    d = client.get("/rag/search", params={"q": "machine learning engineer", "k": 3, "kind": "occupation"}).json()
    assert d["hits"] and all(h["id"] and h["kind"] == "occupation" for h in d["hits"])
    c = client.get("/rag/search", params={"q": "docker containers", "kind": "course"}).json()["hits"]
    assert c and c[0]["meta"]["url"].startswith("http")

def test_extract_evidence_verified():
    r = client.post("/extract/text", json={"resume_text": RESUME, "target_role": "ML Engineer"}).json()
    assert r["skills"]
    for s in r["skills"]:
        assert verify_evidence(RESUME, s["evidence"])

def test_score_identical_across_processes():
    """Guards against set-order / hash-seed nondeterminism (found and fixed during development)."""
    import subprocess
    code = ("import sys,json;sys.path.insert(0,'.');from app.routers.match import run_match;"
            f"print(run_match({RESUME!r},{JD!r})['match_score'])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(s)}, cwd=os.path.join(os.path.dirname(__file__), "..")).stdout.strip().splitlines()[-1]
            for s in (1, 2, 3)}
    assert len(outs) == 1, outs
