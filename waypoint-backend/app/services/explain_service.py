"""Module 3: evidence-linked, explainable matching.

For every job requirement the matcher finds the best supporting resume span, gives a verdict
(met / partial / not_met), and a VERIFIER checks that the quoted evidence really occurs in the
resume text. The final score and the reasoning trace are computed by code from these values,
not written by an LLM. Optional LLM judging (use_llm) is accepted only when its quote verifies.
"""
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.services.embedding_service import get_embedder
from app.services import llm

GENERIC = {"api", "apis", "skills", "skill", "experience", "knowledge", "tools", "tool", "services",
           "service", "framework", "frameworks", "development", "software", "and", "of", "with", "in"}
LEARNER_CUES = re.compile(r"\b(basic|beginner|learning|familiar(?:ity)?|exposure|coursework|currently studying|"
                          r"introductory|academic|self-taught|exploring)\b", re.I)
VERDICT_SCORE = {"met": 1.0, "partial": 0.5, "not_met": 0.0}

# -- resume segmentation ----------------------------------------------------------------
def split_spans(text: str) -> List[Dict]:
    """Split a resume into line/sentence spans (keeping hyphens inside words) with character offsets."""
    spans, cursor = [], 0
    for raw in text.splitlines(keepends=True):
        line_start = cursor; cursor += len(raw)
        line = re.sub(r"^[\s\-\u2022\u25CF\*\u2013>]+", "", raw.strip("\r\n"))
        for part in re.split(r"(?<=[.;])\s+", line):
            t = part.strip(" \t:|")
            if len(t) >= 4:
                start = text.find(t, line_start)
                if start >= 0:
                    spans.append({"text": t, "start": start, "end": start + len(t)})
    return spans

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def verify_evidence(resume_text: str, evidence: str) -> bool:
    """True only if the quoted evidence occurs verbatim (whitespace/case-insensitive) in the resume."""
    return bool(evidence.strip()) and _norm(evidence) in _norm(resume_text)

def is_list_span(text: str) -> bool:
    """A skills-list line (>=4 short comma/semicolon/pipe-separated items) names skills without showing use."""
    parts = [x for x in re.split(r"[,;|]", text) if x.strip()]
    return len(parts) >= 4 and sum(len(x.split()) for x in parts) / len(parts) <= 3

# -- lexical (stem-aware) check ---------------------------------------------------------
def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9\+#\./]+", s.lower())

def lexical_hit(requirement: str, span: str) -> bool:
    """All content tokens of the requirement prefix-match a token of the span ('REST' ~ 'RESTful')."""
    req = [t for t in _tokens(requirement) if t not in GENERIC] or _tokens(requirement)
    toks = _tokens(span)
    def ok(rt):
        return any(t == rt or (len(rt) >= 3 and (t.startswith(rt) or (len(t) >= 4 and rt.startswith(t)))) for t in toks)
    return bool(req) and all(ok(rt) for rt in req)

# -- core matcher -----------------------------------------------------------------------
def match_requirements(resume_text: str, requirements: List[Dict], use_llm: bool = False) -> Tuple[List[Dict], str]:
    emb = get_embedder()
    spans = split_spans(resume_text) or [{"text": resume_text.strip()[:300], "start": 0, "end": min(300, len(resume_text))}]
    names = [r["name"] for r in requirements]
    sim = emb.similarity(names, [s["text"] for s in spans]) if names else np.zeros((0, len(spans)))
    th = emb.thresholds
    results = []
    for i, r in enumerate(requirements):
        lex_idx = [j for j, s in enumerate(spans) if lexical_hit(r["name"], s["text"])]
        if lex_idx:
            j = max(lex_idx, key=lambda x: sim[i, x]); method = "lexical"; score = 1.0
            non_list = [x for x in lex_idx if not is_list_span(spans[x]["text"])]
            if non_list:
                j = max(non_list, key=lambda x: sim[i, x])
            if LEARNER_CUES.search(spans[j]["text"]):
                verdict, reason = "partial", "Requirement term found in resume; learner wording lowers it to partial."
            elif not non_list:
                verdict, reason = "partial", "Skill only appears in a skills list, with no usage context (capped at partial)."
            else:
                verdict, reason = "met", "Requirement term (or a stem variant) found in a sentence showing its use."
        else:
            j = int(np.argmax(sim[i])); method = emb.name; score = float(sim[i, j])
            verdict = "met" if score >= th["met"] else "partial" if score >= th["partial"] else "not_met"
            reason = f"Best semantic match, cosine {score:.2f} (thresholds met>={th['met']}, partial>={th['partial']})."
        sp = spans[j]
        res = {"requirement": r["name"], "weight": r["weight"], "verdict": verdict, "score": VERDICT_SCORE[verdict],
               "evidence": sp["text"][:240] if verdict != "not_met" else "", "evidence_span": [sp["start"], sp["end"]] if verdict != "not_met" else None,
               "similarity": round(score, 3), "method": method, "reason": reason if verdict != "not_met" else "No resume span is similar enough to this requirement."}
        results.append(res)

    mode = "deterministic"
    if use_llm and llm.llm_available() and results:
        mode = "llm+verifier"
        data = llm.chat_json(
            "You judge whether a resume satisfies each job requirement. Respond with JSON only: "
            '{"items":[{"requirement":"...","verdict":"met|partial|not_met","evidence":"<verbatim quote from the resume, max 25 words>"}]}. '
            "Quote the resume exactly; if there is no evidence use not_met and an empty quote.",
            f"Requirements: {names}\n\nResume:\n{resume_text[:6000]}", 1800)
        by = {str(it.get("requirement", "")).lower(): it for it in (data or {}).get("items", [])}
        for res in results:
            it = by.get(res["requirement"].lower())
            if not it:
                continue
            v, ev = it.get("verdict"), str(it.get("evidence", ""))
            if v in VERDICT_SCORE and (v == "not_met" or verify_evidence(resume_text, ev)):
                res.update(verdict=v, score=VERDICT_SCORE[v], evidence=ev if v != "not_met" else "", method="llm",
                           evidence_span=None, reason="LLM verdict accepted: quoted evidence verified in resume.")
            elif v in VERDICT_SCORE:
                res["reason"] += " (LLM verdict rejected: its quote was not found in the resume.)"
                res["llm_rejected"] = True

    for res in results:   # the verifier runs on every item, whatever produced it
        res["evidence_verified"] = verify_evidence(resume_text, res["evidence"]) if res["evidence"] else None
        if res["evidence"] and not res["evidence_verified"]:
            res.update(verdict="not_met", score=0.0, evidence="", evidence_span=None,
                       reason="Evidence failed verification and was discarded.")
    return results, mode

def score_results(results: List[Dict]) -> float:
    tw = sum(r["weight"] for r in results) or 1.0
    return round(100.0 * sum(r["weight"] * r["score"] for r in results) / tw, 1)

def keyword_baseline(resume_text: str, requirements: List[Dict]) -> float:
    """Config A baseline: weighted share of requirement strings occurring literally in the resume."""
    t = _norm(resume_text); tw = sum(r["weight"] for r in requirements) or 1.0
    return round(100.0 * sum(r["weight"] for r in requirements if _norm(r["name"]) in t) / tw, 1)

def document_similarity(resume_text: str, jd: str) -> float:
    emb = get_embedder()
    return round(float(emb.similarity([resume_text[:3000]], [jd[:3000]])[0, 0]), 3)

def build_trace(results: List[Dict], score: float, mode: str, backend: str) -> List[str]:
    c = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICT_SCORE}
    ver = sum(1 for r in results if r["evidence_verified"])
    withev = sum(1 for r in results if r["evidence"])
    tr = [f"[REQ] {len(results)} requirements, weights sum to {sum(r['weight'] for r in results):.2f}",
          f"[MATCH] backend={backend}, mode={mode}",
          f"[VERDICT] met={c['met']} partial={c['partial']} not_met={c['not_met']}",
          f"[VERIFY] {ver}/{withev} evidence quotes verified verbatim in the resume",
          f"[SCORE] sum(weight x verdict) = {score}/100  (met=1, partial=0.5, not_met=0)"]
    rej = [r["requirement"] for r in results if r.get("llm_rejected")]
    if rej:
        tr.append(f"[LLM-REJECT] verdict rejected for: {', '.join(rej)}")
    return tr
