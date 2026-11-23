# WayPoint — explainable, RAG-grounded resume–job matching

Seminar/mini-project prototype aligned with the project synopsis: **NLP + embeddings + retrieval + LLM** for
resume–job matching with **explainable scores**, **skill-gap recommendation** and **fairness/robustness evaluation**.
Research prototype on public/synthetic data; not a production ATS.

## What it does
`POST /match` takes a resume (file or text) and a job description and returns:
1. **Per-requirement verdicts** (met / partial / not met) with a **quoted resume span**; a verifier rejects any quote not literally in the resume.
2. **Match score** computed in code: Σ weight × verdict (met=1, partial=0.5), plus a keyword-only baseline and whole-document cosine similarity.
3. **RAG grounding**: hybrid retrieval (dense + TF-IDF, rank-fused) over O*NET occupations (1,016), the course catalogue (28 modules) and Coursera courses (3,522).
4. **Skill-gap pathway**: gaps → catalogue modules → transitive prerequisites → topological order; courses are retrieved, never invented.
5. **Fairness/robustness** (`POST /match/fairness`): identity (name + gendered words), typo, section-order and keyword-stuffing variants.

## Modules (one per team member)
| Module | Code |
|---|---|
| Semantic matching & embeddings | `services/embedding_service.py` (SBERT, or TF-IDF fallback) |
| RAG-based recruitment | `services/rag_service.py`, `routers/rag.py` |
| Explainable, context-aware matching | `services/explain_service.py`, `services/requirements_service.py` |
| Fairness, robustness, skill gap | `services/fairness_service.py`, `services/graph_service.py`, `eval/run_eval.py` |

## Run
```bash
cd waypoint-backend
pip install -r requirements.txt
pip install -r requirements-sbert.txt      # recommended: real semantic embeddings (downloads a model)
cp .env.example .env                       # GROQ_API_KEY optional; the app runs without it
uvicorn app.main:app --reload --port 8000
# frontend
cd ../waypoint-frontend/neuralpath && npm install && npm run dev
```
Without `sentence-transformers` the app falls back to a **lexical, stem-aware matcher** (the UI shows a notice).
Without a Groq key it runs in **deterministic mode**; with a key, tick “LLM judge” — its verdicts are accepted only if their quotes verify.

## Test and evaluate
```bash
cd waypoint-backend
pytest -q tests            # 11 offline tests (verified evidence, determinism, prerequisite order, fairness, retrieval)
python eval/run_eval.py    # persona x job matrix, keyword baseline vs matcher, fairness + robustness -> eval/results.md
```
`eval/results.md` is a face-validity check on 4 demo personas and 4 **synthetic** JDs; it is not an accuracy benchmark.

## Known limitations
- Verdict thresholds for the SBERT backend (`embedding_service.THRESHOLDS`) are uncalibrated defaults; calibrate on labelled skill pairs.
- Keyword stuffing still raises scores (list-only skills are capped at *partial*, which reduces but does not remove the effect).
- “Hours saved” is measured only against the uncompressed catalogue duration; the 65% partial-skill factor is a heuristic.
- `ml-engine/` (O*NET skill DAG, proficiency estimator) is an offline research tool; the API uses the catalogue DAG. `export_onet_for_rag.py` feeds the RAG index.

## Removed from the earlier prototype (outside the synopsis scope)
ATS resume scorer, adaptive skill tests, onboarding quiz, certificate page, demo/skill-graph pages, LLM-generated learning modules, and the
unmeasured claims (“0% hallucination”, “63% time reduction”, Dijkstra).
