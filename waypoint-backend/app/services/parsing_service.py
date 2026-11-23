"""Resume parsing: dictionary NER + optional LLM proficiency classification.

Every skill must carry an evidence quote that is verified to occur in the resume; skills whose
quote fails verification are dropped. The trace is generated from counts computed here.
"""
import re
from typing import List, Tuple
from app.models import ExtractedSkill, ProficiencyLevel
from app.services.ner_service import extract_candidate_skills
from app.services.explain_service import verify_evidence, LEARNER_CUES
from app.services import llm

STRONG = re.compile(r"\b(built|developed|led|designed|implemented|deployed|architected|managed|trained|shipped|optimi[sz]ed|owned)\b", re.I)

LLM_SYSTEM = """You classify resume skills. For each skill from the provided candidate list, respond with JSON only:
{"skills":[{"name":"...","proficiency":"proficient|partial","confidence":0.0-1.0,"evidence":"<verbatim quote from the resume, max 15 words>"}]}
proficient = used professionally/in projects with concrete outcomes; partial = coursework, basic, learning, brief mention.
Quote the resume exactly. Do not add skills that are not in the candidate list."""

def extract_skills_from_resume(resume_text: str, target_role: str = "") -> Tuple[List[ExtractedSkill], List[str]]:
    cands, years = extract_candidate_skills(resume_text)
    trace = [f"[NER] dictionary scan found {len(cands)} candidate skills"
             + (f"; year hints for {', '.join(list(years)[:4])}" if years else "")]
    skills: List[ExtractedSkill] = []
    llm_by_name = {}
    if cands and llm.llm_available():
        data = llm.chat_json(LLM_SYSTEM, f"Candidate skills: {[c[0] for c in cands]}\n\nResume:\n{resume_text[:6000]}", 1800)
        llm_by_name = {str(s.get("name", "")).lower(): s for s in (data or {}).get("skills", [])}
        trace.append(f"[LLM] classified {len(llm_by_name)} skills" if llm_by_name else "[LLM] unavailable or failed; using rules")
    dropped = 0
    for name, ctx in cands:
        item = llm_by_name.get(name.lower())
        if item and verify_evidence(resume_text, str(item.get("evidence", ""))):
            prof = ProficiencyLevel.PROFICIENT if item.get("proficiency") == "proficient" else ProficiencyLevel.PARTIAL
            conf, ev = float(item.get("confidence", 0.6)), str(item["evidence"])
        else:
            if item:
                dropped += 1       # LLM quote did not verify: fall back to the NER context, which is verbatim
            if LEARNER_CUES.search(ctx):
                prof, conf = ProficiencyLevel.PARTIAL, 0.7
            elif years.get(name, 0) >= 2 or STRONG.search(ctx):
                prof, conf = ProficiencyLevel.PROFICIENT, 0.7
            else:
                prof, conf = ProficiencyLevel.PARTIAL, 0.55
            ev = ctx
        if verify_evidence(resume_text, ev):
            skills.append(ExtractedSkill(name=name, proficiency=prof, confidence=round(conf, 2), evidence=ev))
    if dropped:
        trace.append(f"[VERIFY] {dropped} LLM evidence quotes failed verification; rule-based evidence used instead")
    trace.append(f"[FINAL] {len(skills)} skills, all with evidence verified in the resume")
    return skills, trace
