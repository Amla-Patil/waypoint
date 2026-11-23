from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ProficiencyLevel(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    PROFICIENT = "proficient"


class ExtractedSkill(BaseModel):
    name: str
    proficiency: ProficiencyLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(..., description="Quote from the resume (verified to occur verbatim)")


class ExtractResponse(BaseModel):
    skills: List[ExtractedSkill]
    target_role: str
    reasoning_trace: List[str]


class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
    use_llm: bool = False


class RequirementResult(BaseModel):
    requirement: str
    weight: float
    verdict: str                      # met | partial | not_met
    score: float
    evidence: str
    evidence_span: Optional[List[int]] = None
    evidence_verified: Optional[bool] = None
    similarity: float
    method: str
    reason: str
    llm_rejected: Optional[bool] = None


class PathwayModuleOut(BaseModel):
    module_id: str
    title: str
    hours: int
    full_hours: int
    priority: str                     # CORE GAP | REFRESH | PREREQUISITE
    prerequisites: List[str]
    why_included: str
    course_title: Optional[str] = None
    course_url: Optional[str] = None
    course_provider: Optional[str] = None
    course_similarity: Optional[float] = None
    course_id: Optional[str] = None


class Grounding(BaseModel):
    code: str
    title: str
    similarity: float
    top_skills: List[str]


class MatchResponse(BaseModel):
    role_title: str
    match_score: float
    baseline_keyword_score: float
    semantic_similarity: float
    embedding_backend: str
    mode: str
    requirements: List[RequirementResult]
    skill_gaps: List[str]
    partial_skills: List[str]
    met_skills: List[str]
    grounding: List[Grounding]
    modules: List[PathwayModuleOut]
    resources: list
    total_hours: int
    full_hours: int
    hours_saved: int
    reasoning_trace: List[str]


class CatalogModule(BaseModel):
    id: str
    title: str
    domain: str
    hours: int
    description: str
    prerequisites: List[str]
    tags: List[str]
