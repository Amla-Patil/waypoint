"""Module 4b: counterfactual fairness and robustness checks on the matcher.

Identity variants change only name and gendered wording; robustness variants change typography,
section order, or append keyword stuffing. A fair, robust matcher should show near-zero score
shift for identity/typo/order variants; stuffing is a known weakness we report, not hide.
"""
import random, re
from typing import Dict, List
from app.services.explain_service import match_requirements, score_results

IDENTITY_VARIANTS = [
    {"group": "A", "name": "Rahul Sharma",   "gender": "m"},
    {"group": "B", "name": "Priya Sharma",   "gender": "f"},
    {"group": "C", "name": "Mohammed Khan",  "gender": "m"},
    {"group": "D", "name": "Anjali Iyer",    "gender": "f"},
]
_PRON = {"m": {"she": "he", "her": "his", "hers": "his", "ms.": "mr.", "mrs.": "mr."},
         "f": {"he": "she", "his": "her", "him": "her", "mr.": "ms."}}

def _swap_words(text: str, gender: str) -> str:
    table = _PRON[gender]
    def rep(m):
        w = m.group(0); r = table[w.lower()]
        return r.capitalize() if w[0].isupper() else r
    return re.sub(r"\b(" + "|".join(re.escape(k) for k in table) + r")(?=\W|$)", rep, text, flags=re.I)

def identity_variant(text: str, v: Dict) -> str:
    lines = text.strip().split("\n")
    first = lines[0].strip()
    body = lines[1:] if (len(first.split()) <= 4 and not re.search(r"\d|@", first)) else lines
    return _swap_words(v["name"] + "\n" + "\n".join(body), v["gender"])

def typo_variant(text: str, rate: float = 0.03, seed: int = 7) -> str:
    rnd = random.Random(seed); out = []
    for w in text.split(" "):
        if len(w) > 4 and rnd.random() < rate:
            i = rnd.randrange(1, len(w) - 2); w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)

def shuffle_variant(text: str, seed: int = 7) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if len(blocks) < 3:
        lines = text.strip().split("\n"); head, rest = lines[:1], lines[1:]
        random.Random(seed).shuffle(rest); return "\n".join(head + rest)
    head, rest = blocks[0], blocks[1:]
    random.Random(seed).shuffle(rest); return "\n\n".join([head] + rest)

def stuffing_variant(text: str, requirements: List[Dict]) -> str:
    return text.rstrip() + "\n\nSkills: " + ", ".join(r["name"] for r in requirements) + "\n"

def _score(text: str, reqs: List[Dict]) -> float:
    return score_results(match_requirements(text, reqs)[0])

def run_checks(resume: str, reqs: List[Dict], eps: float = 2.0) -> Dict:
    base = _score(resume, reqs)
    ident = [{"group": v["group"], "name": v["name"], "score": _score(identity_variant(resume, v), reqs)} for v in IDENTITY_VARIANTS]
    shifts = [abs(i["score"] - base) for i in ident]
    robust = []
    for label, fn in [("typos (3% of words)", typo_variant), ("section order shuffled", shuffle_variant),
                      ("keyword stuffing appended", lambda t: stuffing_variant(t, reqs))]:
        s = _score(fn(resume), reqs)
        robust.append({"variant": label, "score": s, "shift": round(s - base, 1)})
    stuff = robust[-1]["shift"]
    return {"base_score": base, "tolerance": eps, "identity": ident,
            "identity_max_shift": round(max(shifts), 1), "identity_pass": max(shifts) <= eps,
            "robustness": robust,
            "stuffing_vulnerable": stuff > eps,
            "notes": ["Identity variants change only name and gendered words.",
                      "Keyword stuffing is not mitigated in this version: a skills line is indistinguishable from a legitimate skills section.",
                      "This check covers the matcher you are running (deterministic mode shows ~0 identity shift by construction; "
                      "run it with use_llm=true to test an LLM judge)."]}
