# Setup

**Backend** (Python 3.11+): `cd waypoint-backend && pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000`
Optional semantic embeddings: `pip install -r requirements-sbert.txt` (first start downloads `all-MiniLM-L6-v2`; the RAG index is cached in `app/.cache`).
Optional LLM: put `GROQ_API_KEY=...` in `waypoint-backend/.env` (never commit it).

**Frontend**: `cd waypoint-frontend/neuralpath && npm install && npm run dev` (set `VITE_API_URL` if the API is not on `localhost:8000`).

**Docker**: `docker compose up --build`

**Check**: `GET /health` shows the active embedding backend and whether an LLM is available. `pytest -q tests` runs the offline tests.

| Problem | Fix |
|---|---|
| UI banner “running without the sentence-embedding model” | `pip install -r requirements-sbert.txt` and restart |
| 422 “No recognisable skill requirements” | List skills in the JD, e.g. `Required: Python, SQL` |
| `onet_occupations.json missing` | `python ml-engine/export_onet_for_rag.py` |
