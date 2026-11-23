"""Face-validity + fairness evaluation on the demo personas x synthetic job descriptions.
No human labels exist for these pairs, so this is NOT an accuracy benchmark: it checks that
(1) each persona scores highest for its own role, for the keyword baseline (config A) and the
full matcher (config B-D in the active backend), and (2) identity/robustness variants do not move scores.
Run from waypoint-backend/:  python eval/run_eval.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from app.services import explain_service as ex, fairness_service as fs
from app.services.embedding_service import get_embedder
from app.services.requirements_service import extract_requirements

HERE = os.path.dirname(__file__)
personas = json.load(open(os.path.join(HERE, "..", "..", "ml-engine", "onet_data", "demo_profiles.json")))
jobs = json.load(open(os.path.join(HERE, "sample_jobs.json")))
ids = [p["id"] for p in personas]
assert ids == [j["id"] for j in jobs], "personas and jobs must align"
reqs = {j["id"]: extract_requirements(j["jd"]) for j in jobs}

A = np.zeros((4, 4)); B = np.zeros((4, 4))
for i, p in enumerate(personas):
    for k, j in enumerate(jobs):
        A[i, k] = ex.keyword_baseline(p["resume_text"], reqs[j["id"]])
        B[i, k] = ex.score_results(ex.match_requirements(p["resume_text"], reqs[j["id"]])[0])
def top1(M): return float(np.mean([int(np.argmax(M[i]) == i) for i in range(4)]))

CUT = 30.0; rates = {v["group"]: [] for v in fs.IDENTITY_VARIANTS}; shifts = []
for i, p in enumerate(personas):
    rq = reqs[ids[i]]; base = ex.score_results(ex.match_requirements(p["resume_text"], rq)[0])
    for v in fs.IDENTITY_VARIANTS:
        s = ex.score_results(ex.match_requirements(fs.identity_variant(p["resume_text"], v), rq)[0])
        rates[v["group"]].append(s >= CUT); shifts.append(abs(s - base))
sel = {g: float(np.mean(r)) for g, r in rates.items()}
di = (min(sel.values()) / max(sel.values())) if max(sel.values()) > 0 else float("nan")
rob = {"typos": [], "shuffle": [], "stuffing": []}
for i, p in enumerate(personas):
    rq = reqs[ids[i]]; base = ex.score_results(ex.match_requirements(p["resume_text"], rq)[0])
    for key, fn in (("typos", fs.typo_variant), ("shuffle", fs.shuffle_variant), ("stuffing", lambda t: fs.stuffing_variant(t, rq))):
        rob[key].append(ex.score_results(ex.match_requirements(fn(p["resume_text"]), rq)[0]) - base)

out = [f"# Evaluation run (backend = {get_embedder().name})", "",
       "Synthetic JDs, demo personas, no human labels: face-validity only.", "",
       "## Score matrix (rows = persona, cols = job), full matcher", "", "| persona | " + " | ".join(ids) + " |", "|---|" + "---|" * 4]
for i in range(4): out.append(f"| {ids[i]} | " + " | ".join(f"{B[i,k]:.1f}" for k in range(4)) + " |")
out += ["", "## Config A (keyword baseline) vs full matcher", "", "| metric | A: keyword | full matcher |", "|---|---|---|",
        f"| top-1 own-role accuracy | {top1(A):.2f} | {top1(B):.2f} |",
        f"| mean score on own role | {np.mean([A[i,i] for i in range(4)]):.1f} | {np.mean([B[i,i] for i in range(4)]):.1f} |",
        f"| mean score on other roles | {np.mean([A[i,k] for i in range(4) for k in range(4) if i!=k]):.1f} | {np.mean([B[i,k] for i in range(4) for k in range(4) if i!=k]):.1f} |",
        "", "## Fairness (identity variants: name + gendered words)", "",
        f"- max |score shift| over {len(shifts)} variants: {max(shifts):.2f}; mean {np.mean(shifts):.2f}",
        f"- selection rate at cutoff {CUT:.0f} per group: " + ", ".join(f"{g}={r:.2f}" for g, r in sel.items()),
        f"- disparate-impact ratio (min/max, four-fifths rule threshold 0.8): {di:.2f}", "",
        "## Robustness (mean score change vs original, own-role pairs)", ""]
for k, v in rob.items(): out.append(f"- {k}: mean {np.mean(v):+.1f}, max {max(v):+.1f}, min {min(v):+.1f}")
open(os.path.join(HERE, "results.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
