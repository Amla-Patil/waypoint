"""Skill-gap pathway (Module 4): gaps -> catalogue modules -> prerequisites -> topological order.

Unlike the earlier prototype, this service (a) adds transitive prerequisites, (b) orders modules by
topological sort of the prerequisite DAG, (c) never asks an LLM to invent modules, and (d) reports
hours saved only against a measurable baseline: the full (uncompressed) catalogue duration.
"""
import json, os
from typing import Dict, List, Optional
import networkx as nx

from app.services import rag_service
from app.services.embedding_service import get_embedder

_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "course_catalog.json")
with open(_CATALOG_PATH, encoding="utf-8") as f:
    CATALOG: List[Dict] = json.load(f)
CATALOG_BY_ID = {m["id"]: m for m in CATALOG}
_ORDER = {m["id"]: i for i, m in enumerate(CATALOG)}
CATALOG_BY_TAGS: Dict[str, List[str]] = {}
for _m in CATALOG:
    for _t in _m["tags"]:
        CATALOG_BY_TAGS.setdefault(_t.lower(), []).append(_m["id"])

PARTIAL_FACTOR = 0.65   # heuristic: a partially-known skill needs 65% of module hours (not validated)
MODULE_MIN_SIM = 0.30   # minimum retrieval similarity to accept a module for an unmapped skill

def build_prerequisite_dag() -> nx.DiGraph:
    G = nx.DiGraph()
    for m in CATALOG:
        G.add_node(m["id"])
        for p in m["prerequisites"]:
            G.add_edge(p, m["id"])
    assert nx.is_directed_acyclic_graph(G), "course catalogue prerequisites contain a cycle"
    return G

DAG = build_prerequisite_dag()

def skill_to_module_ids(skill: str) -> List[str]:
    """Tag lookup first; if nothing matches, ask the retriever for the closest catalogue module."""
    s = skill.lower()
    hit = set(CATALOG_BY_TAGS.get(s, []))
    for tag, ids in CATALOG_BY_TAGS.items():
        if s in tag or tag in s:
            hit.update(ids)
    if hit:
        return sorted(hit, key=_ORDER.get)
    for h in rag_service.retrieve(skill, k=1, kinds=["module"]):
        if h["similarity"] >= MODULE_MIN_SIM:
            return [h["id"]]
    return []

def attach_course(module: Dict) -> Dict:
    q = f"{module['title']} {' '.join(module['tags'])}"
    hits = rag_service.retrieve(q, k=1, kinds=["course"])
    min_sim = get_embedder().thresholds["met"] * 0.9      # below this the link would be a guess: show none
    if hits and hits[0]["similarity"] >= min_sim:
        h = hits[0]
        return {"course_title": h["title"], "course_url": h["meta"]["url"], "course_provider": f"Coursera / {h['meta']['university']}",
                "course_similarity": h["similarity"], "course_id": h["id"]}
    return {"course_title": None, "course_url": None, "course_provider": None, "course_similarity": None, "course_id": None}

def build_pathway(gaps: List[str], partials: List[str], proficient: List[str]) -> Dict:
    known_tags = {p.lower() for p in proficient}
    direct: Dict[str, str] = {}           # module_id -> priority label
    unmapped: List[str] = []
    trace: List[str] = []
    for skill, label in [(g, "CORE GAP") for g in gaps] + [(p, "REFRESH") for p in partials]:
        ids = skill_to_module_ids(skill)
        if not ids:
            unmapped.append(skill); continue
        for mid in ids[:2]:
            direct.setdefault(mid, label)
            if label == "CORE GAP":
                direct[mid] = "CORE GAP"
    wanted = set(direct)
    for mid in list(direct):
        for anc in nx.ancestors(DAG, mid):
            if set(t.lower() for t in CATALOG_BY_ID[anc]["tags"]) & known_tags:
                continue                   # candidate already proficient in this prerequisite
            wanted.add(anc)
    sub = DAG.subgraph(wanted)
    order = list(nx.lexicographical_topological_sort(sub, key=lambda n: _ORDER[n]))
    modules, full, total = [], 0, 0
    for mid in order:
        m = CATALOG_BY_ID[mid]
        prio = direct.get(mid, "PREREQUISITE")
        hours = round(m["hours"] * (PARTIAL_FACTOR if prio == "REFRESH" else 1.0))
        full += m["hours"]; total += hours
        mod = {"module_id": mid, "title": m["title"], "hours": hours, "full_hours": m["hours"], "priority": prio,
               "prerequisites": [p for p in m["prerequisites"] if p in wanted],
               "why_included": {"CORE GAP": "Required skill is missing from the resume.",
                                "REFRESH": f"Skill is only partly evidenced; hours reduced to {int(PARTIAL_FACTOR*100)}% (heuristic).",
                                "PREREQUISITE": "Prerequisite of a later module in this pathway."}[prio]}
        mod.update(attach_course(m))
        modules.append(mod)
    resources = []
    for s in unmapped:
        resources.append({"skill": s, "courses": [
            {"title": h["title"], "url": h["meta"]["url"], "similarity": h["similarity"]}
            for h in rag_service.retrieve(s, k=3, kinds=["course"])]})
    pos = {m["module_id"]: i for i, m in enumerate(modules)}
    assert all(pos[p] < pos[m["module_id"]] for m in modules for p in m["prerequisites"]), "prerequisite order violated"
    trace.append(f"[PATHWAY] {len(direct)} directly needed modules + {len(wanted) - len(direct)} prerequisites, topologically ordered")
    trace.append(f"[HOURS] {total}h planned vs {full}h for the same modules uncompressed; saving comes only from partial-skill compression")
    if unmapped:
        trace.append(f"[RESOURCES] no catalogue module for: {', '.join(unmapped)} - retrieved Coursera courses instead")
    return {"modules": modules, "total_hours": total, "full_hours": full, "hours_saved": full - total,
            "resources": resources, "trace": trace}
