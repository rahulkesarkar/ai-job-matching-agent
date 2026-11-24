from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.candidate_profile import (
    CandidateProfile,
    extract_skills,
)

logger = logging.getLogger(__name__)


class LLMNarrative(BaseModel):
    """Constrained narrative returned by the local language model."""

    model_config = ConfigDict(extra="forbid")

    match_summary: str = Field(min_length=10, max_length=500)
    recommendation: str = Field(min_length=10, max_length=500)


class JobAnalysisResult(BaseModel):
    """Validated job analysis returned by the analysis service."""

    job_id: str
    job_title: str
    company: str
    location: str

    match_summary: str
    required_skills: list[str]
    preferred_skills: list[str]
    matched_skills: list[str]
    skill_gaps: list[str]
    preferred_skill_gaps: list[str]
    required_skill_match_count: int
    required_skill_count: int
    required_skill_coverage: float | None

    candidate_evidence: list[str]
    job_evidence: list[str]

    recommendation: str

    provider: Literal["ollama", "deterministic_fallback"]
    model: str
    used_fallback: bool



def _safe_text(value: Any, limit: int = 350) -> str:
    """Redact contact details while preserving dates and grades."""
    if not isinstance(value, str):
        return ""

    text = value.strip()

    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[redacted email]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[redacted link]",
        text,
        flags=re.IGNORECASE,
    )

    def redact_phone(match):
        value = match.group(0)
        digit_count = sum(character.isdigit() for character in value)

        # Indian mobile numbers and international phone numbers.
        # Short numeric ranges, years and grades remain untouched.
        if 10 <= digit_count <= 15:
            return "[redacted phone]"

        return value

    text = re.sub(
        r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)",
        redact_phone,
        text,
    )

    return re.sub(r"\s+", " ", text)[:limit]

def _candidate_evidence(
    profile: CandidateProfile,
    relevant_skills: set[str] | None = None,
) -> list[str]:
    """Prioritize resume evidence relevant to the selected job."""
    evidence = []
    seen = set()
    relevant_skills = relevant_skills or set()

    domain_cues = {
        "dashboard", "sales", "analysis", "analytics", "report",
        "visualization", "data cleaning", "kpi", "metrics",
        "business intelligence", "forecast", "business analyst",
        "stakeholder", "data", "excel",
    }

    section_budgets = {
        "skills": 1,
        "experience": 2,
        "projects": 2,
        "certifications": 1,
    }

    for section_name, budget in section_budgets.items():
        content = profile.sections.get(section_name, "")
        candidates = []

        for position, line in enumerate(content.splitlines()):
            cleaned = _safe_text(line, limit=220)
            if not cleaned:
                continue

            detected = set(extract_skills(cleaned))

            if section_name == "skills" and not detected:
                continue

            if (
                section_name in {"experience", "projects"}
                and len(cleaned) < 25
                and not detected
            ):
                continue

            if section_name == "certifications" and len(cleaned) < 10:
                continue

            skill_overlap = len(detected.intersection(relevant_skills))
            lower_line = cleaned.casefold()
            cue_overlap = sum(
                1 for cue in domain_cues if cue in lower_line
            )

            candidates.append(
                (
                    skill_overlap,
                    cue_overlap,
                    len(detected),
                    -position,
                    cleaned,
                )
            )

        candidates.sort(reverse=True)

        for _, _, _, _, cleaned in candidates[:budget]:
            item = f"{section_name.title()}: {cleaned}"

            if item.casefold() not in seen:
                evidence.append(item)
                seen.add(item.casefold())

    for qualification in profile.education_evidence[:1]:
        cleaned = _safe_text(qualification, limit=220)
        if cleaned:
            item = f"Education: {cleaned}"
            if item.casefold() not in seen:
                evidence.append(item)
                seen.add(item.casefold())

    return evidence


REQUIRED_SKILL_CUE = re.compile(
    r"\b(?:requirements?|required|mandatory|essential|must[- ]have|"
    r"minimum qualifications?|proficiency in|proficient in|"
    r"experience with|hands[- ]on experience with|"
    r"strong knowledge of|knowledge of|skills include|"
    r"technical skills|key skills|must be able to|"
    r"are required|is required)\b",
    re.IGNORECASE,
)

PREFERRED_SKILL_CUE = re.compile(
    r"\b(?:preferred|nice[- ]to[- ]have|good[- ]to[- ]have|"
    r"desirable|bonus|an advantage|would be (?:a )?plus|"
    r"is a plus|ideally|preferred qualifications?)\b",
    re.IGNORECASE,
)

REQUIRED_HEADINGS = {
    "requirements",
    "requirement",
    "qualifications",
    "required skills",
    "technical skills",
    "key skills",
    "required qualifications",
    "minimum qualifications",
    "what you will need",
    "what you ll need",
    "what you will bring",
    "what you ll bring",
    "must have",
}

PREFERRED_HEADINGS = {
    "preferred",
    "preferred skills",
    "preferred qualifications",
    "desired skills",
    "nice to have",
    "good to have",
    "bonus skills",
}

NEUTRAL_HEADINGS = {
    "about us",
    "about the company",
    "company overview",
    "who we are",
    "responsibilities",
    "what you will do",
    "what you ll do",
    "benefits",
    "job description",
    "equal opportunity",
}

REQUIRED_HEADINGS.update({
    "skills required",
    "technical skills required",
    "required skills and qualifications",
    "skills and qualifications",
    "key requirements",
})

NEUTRAL_HEADINGS.update({
    "experience",
    "qualification",
    "duties responsibilities",
    "duties and responsibilities",
    "job description",
    "role overview",
    "about the role",
})




def _extract_job_skill_requirements(
    description: str,
) -> tuple[list[str], list[str]]:
    """
    Extract recognized skills from required and preferred sections.

    Handles heading variations such as "Skills Required" and
    ignores incidental mentions in company descriptions.
    """
    required = []
    preferred = []
    context = "neutral"

    for raw_line in description.splitlines():
        line = re.sub(r"^[\s•*\-]+", "", raw_line).strip()

        if not line:
            continue

        heading_match = re.match(
            r"^([^:]{2,65}):\s*(.*)$",
            line,
        )

        if heading_match:
            heading = heading_match.group(1)
            content = heading_match.group(2)
        else:
            heading = line.rstrip(":").strip()
            content = ""

        normalized = re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z ]", " ", heading.lower()),
        ).strip()

        if normalized in REQUIRED_HEADINGS:
            context = "required"
            if not content:
                continue

        elif normalized in PREFERRED_HEADINGS:
            context = "preferred"
            if not content:
                continue

        elif normalized in NEUTRAL_HEADINGS:
            context = "neutral"
            if not content:
                continue

        else:
            # Ordinary content lines inherit the current section.
            if not content:
                content = line

        segments = re.split(
            r"(?<=[.!?])\s+|;\s*",
            content,
        )

        for segment in segments:
            if PREFERRED_SKILL_CUE.search(segment):
                bucket = "preferred"
            elif REQUIRED_SKILL_CUE.search(segment):
                bucket = "required"
            else:
                bucket = context

            if bucket == "neutral":
                continue

            destination = (
                required if bucket == "required" else preferred
            )

            for skill in extract_skills(segment):
                if skill not in destination:
                    destination.append(skill)

    preferred = [
        skill for skill in preferred
        if skill not in required
    ]

    return required, preferred

def _job_evidence(
    description: str,
    relevant_skills: set[str],
) -> list[str]:
    """Collect job-description lines supporting extracted requirements."""
    evidence = []

    for line in description.splitlines():
        cleaned = _safe_text(line, limit=220)

        if not cleaned:
            continue

        detected = set(extract_skills(cleaned))

        if detected.intersection(relevant_skills):
            if cleaned not in evidence:
                evidence.append(cleaned)

        if len(evidence) >= 4:
            break

    return evidence

def _job_value(job: Mapping[str, Any], field: str) -> str:
    """Safely read a job field from a mapping or Pandas Series."""
    value = job.get(field)

    if value is None:
        return ""

    return str(value).strip()




def _deterministic_narrative(
    matched_skills: list[str],
    skill_gaps: list[str],
    preferred_skill_gaps: list[str] | None = None,
    required_skills: list[str] | None = None,
) -> tuple[str, str]:
    """Generate a factual explanation without an LLM."""
    preferred_skill_gaps = preferred_skill_gaps or []
    required_skills = required_skills or []

    if not required_skills:
        summary = (
            "The requirement extractor did not identify a clear list "
            "of required technical skills in this posting."
        )
        recommendation = (
            "Review the complete job description manually. "
            "A mandatory skill-gap conclusion cannot be made "
            "from the extracted requirements."
        )

    elif matched_skills and skill_gaps:
        summary = (
            "The resume explicitly lists "
            + ", ".join(matched_skills[:8])
            + ", which also appear in the required job skills."
        )
        recommendation = (
            "Required skills not detected in the resume: "
            + ", ".join(skill_gaps[:8])
            + ". Add evidence only for skills you genuinely possess."
        )

    elif matched_skills:
        summary = (
            "The resume explicitly lists "
            + ", ".join(matched_skills[:8])
            + ", covering the required skills identified by this "
            "rule-based extraction."
        )
        recommendation = (
            "No required-skill gaps were identified by the current "
            "extractor. Review the complete posting for other "
            "experience, qualification and language requirements."
        )

    elif skill_gaps:
        summary = (
            "No overlap was detected between the recognized candidate "
            "skills and the required skills in this job posting."
        )
        recommendation = (
            "Required skills not detected in the resume: "
            + ", ".join(skill_gaps[:8])
            + ". Review the requirements and provide additional "
            "evidence only where accurate."
        )

    else:
        summary = (
            "Required skills were extracted, but the candidate-skill "
            "comparison did not return a complete result."
        )
        recommendation = (
            "Review the candidate profile and job requirements manually. "
            "No reliable skill-gap conclusion is available."
        )

    if preferred_skill_gaps:
        recommendation += (
            " Preferred skills not detected: "
            + ", ".join(preferred_skill_gaps[:8])
            + ". These are not classified as mandatory."
        )

    return summary, recommendation


def _reject_unsupported_candidate_skill_claims(
    narrative: LLMNarrative,
    candidate_skills: list[str],
    skill_gaps: list[str],
) -> None:
    """
    Reject affirmative claims that the candidate demonstrates a missing skill.

    A missing skill may be mentioned neutrally or as a requirement, but the
    narrative must not attribute it to the candidate without evidence.
    """
    text = (
        narrative.match_summary
        + " "
        + narrative.recommendation
    )

    sentences = re.split(r"(?<=[.!?])\s+", text)

    candidate_cue = re.compile(
        r"\b(?:candidate|applicant|resume|résumé|cv)\b",
        re.IGNORECASE,
    )

    positive_cue = re.compile(
        r"\b(?:demonstrates?|has|possesses?|proficient|experienced|"
        r"skilled|knows?|expertise|includes?|showcases?|"
        r"experience\s+(?:with|in))\b",
        re.IGNORECASE,
    )

    negative_cue = re.compile(
        r"\b(?:not|no|never|without|lacks?|missing|absent|"
        r"does\s+not|doesn't|has\s+not|hasn't|"
        r"not\s+yet|undemonstrated|not\s+demonstrated|"
        r"not\s+listed|not\s+detected|not\s+shown)\b",
        re.IGNORECASE,
    )

    gaps = set(skill_gaps)

    for sentence in sentences:
        sentence_skills = set(extract_skills(sentence))

        for gap in gaps.intersection(sentence_skills):
            if not candidate_cue.search(sentence):
                continue

            if negative_cue.search(sentence):
                continue

            if positive_cue.search(sentence):
                raise ValueError(
                    "The model attributed an unsupported skill "
                    f"to the candidate: {gap}"
                )


def _generate_narrative(
    candidate_skills: list[str],
    candidate_evidence: list[str],
    job_title: str,
    company: str,
    job_skills: list[str],
    required_job_skills: list[str],
    preferred_job_skills: list[str],
    job_evidence: list[str],
    matched_skills: list[str],
    skill_gaps: list[str],
    preferred_skill_gaps: list[str],
) -> tuple[str, str, bool]:
    """Generate a concise explanation using local Ollama inference."""
    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).rstrip("/")

    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "320"))
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only JSON matching the supplied schema. Keep each field concise and factual, with at most two short sentences. Use only the supplied candidate and job evidence. Never invent candidate skills, qualifications, projects or experience. Do not claim a missing skill is demonstrated. A missing skill means not detected in the resume, not proven absent. Do not use numbered lists or Markdown. Do not rate overall job fit or describe alignment as strong, excellent, ideal or similar. Python calculates skill coverage separately.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "candidate_skills": candidate_skills,
                        "candidate_evidence": candidate_evidence,
                        "job_title": job_title,
                        "company": company,
                        "job_skills": job_skills,
                        "required_job_skills": required_job_skills,
                        "preferred_job_skills": preferred_job_skills,
                        "job_evidence": job_evidence,
                        "computed_matched_skills": matched_skills,
                        "computed_skill_gaps": skill_gaps,
                        "computed_preferred_skill_gaps": preferred_skill_gaps,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "format": LLMNarrative.model_json_schema(),
        "think": False,
        "stream": False,
        "keep_alive": 0,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0,
        },
    }

    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        response.raise_for_status()

        data = response.json()

        if data.get("done_reason") == "length":
            raise ValueError(
                "The local model reached its output-token limit."
            )

        content = data.get("message", {}).get("content", "")
        narrative = LLMNarrative.model_validate_json(content)
        overconfident_match_claim = re.compile(
            r"\b(?:(?:strong|excellent|exceptional|outstanding|perfect|"
            r"ideal|great)\s+(?:overall\s+)?(?:match|alignment|fit)|"
            r"highly\s+(?:aligned|suitable))\b",
            re.IGNORECASE,
        )

        if overconfident_match_claim.search(narrative.match_summary):
            raise ValueError(
                "The model made an uncalibrated qualitative match claim."
            )


        # Validate affirmative claims against candidate evidence.
        _reject_unsupported_candidate_skill_claims(
            narrative=narrative,
            candidate_skills=candidate_skills,
            skill_gaps=skill_gaps,
        )

        return (
            narrative.match_summary,
            narrative.recommendation,
            True,
        )

    except (
        httpx.HTTPError,
        ValueError,
        ValidationError,
        KeyError,
        TypeError,
    ) as exc:
        logger.warning(
            "Local LLM analysis unavailable; using deterministic fallback: %s",
            exc,
        )

        summary, recommendation = _deterministic_narrative(
            matched_skills,
            skill_gaps,
            preferred_skill_gaps,
            required_job_skills,
        )

        return summary, recommendation, False


def analyze_job(
    profile: CandidateProfile,
    job: Mapping[str, Any],
) -> JobAnalysisResult:
    """
    Analyze one job against a structured candidate profile.

    Skill matches and gaps are computed deterministically in Python.
    The local LLM only generates the explanatory narrative.
    """
    if not isinstance(profile, CandidateProfile):
        raise TypeError("profile must be a CandidateProfile.")

    job_id = _job_value(job, "id")
    job_title = _job_value(job, "title") or "Untitled position"
    company = _job_value(job, "company") or "Unspecified company"
    location = _job_value(job, "location") or "Unspecified location"
    description = _job_value(job, "description")

    if not job_id:
        raise ValueError("The selected job must have an ID.")

    if not description:
        raise ValueError("The selected job has no description to analyze.")

    candidate_skills = list(dict.fromkeys(profile.skills))

    required_skills, preferred_skills = (
        _extract_job_skill_requirements(description)
    )

    job_skills = list(dict.fromkeys(
        required_skills + preferred_skills
    ))

    candidate_skill_set = set(candidate_skills)

    matched_skills = [
        skill for skill in required_skills
        if skill in candidate_skill_set
    ]

    skill_gaps = [
        skill for skill in required_skills
        if skill not in candidate_skill_set
    ]

    preferred_skill_gaps = [
        skill for skill in preferred_skills
        if skill not in candidate_skill_set
    ]

    required_skill_match_count = len(matched_skills)
    required_skill_count = len(required_skills)

    required_skill_coverage = (
        required_skill_match_count / required_skill_count
        if required_skill_count > 0
        else None
    )

    candidate_evidence = _candidate_evidence(
        profile,
        relevant_skills=set(job_skills),
    )
    job_evidence = _job_evidence(
        description,
        set(job_skills),
    )

    summary, _llm_recommendation, llm_used = _generate_narrative(
        candidate_skills=candidate_skills,
        candidate_evidence=candidate_evidence,
        job_title=job_title,
        company=company,
        job_skills=job_skills,
        required_job_skills=required_skills,
        preferred_job_skills=preferred_skills,
        job_evidence=job_evidence,
        matched_skills=matched_skills,
        skill_gaps=skill_gaps,
        preferred_skill_gaps=preferred_skill_gaps,
    )

    # Python owns the final recommendation and canonical skill-gap names.
    deterministic_summary, recommendation = _deterministic_narrative(
        matched_skills,
        skill_gaps,
        preferred_skill_gaps,
        required_skills,
    )

    if not llm_used:
        summary = deterministic_summary

    if required_skill_coverage is not None:
        percentage = round(required_skill_coverage * 100)
        coverage_statement = (
            f"Recognized required-skill coverage: "
            f"{required_skill_match_count}/{required_skill_count} "
            f"({percentage}%). This measures extracted skill overlap "
            "only, not overall job fit or hiring probability."
        )
    else:
        coverage_statement = (
            "Required-skill coverage is unavailable because no clear "
            "required-skill list was extracted from the posting."
        )

    summary = f"{coverage_statement} {summary}"

    return JobAnalysisResult(
        job_id=job_id,
        job_title=job_title,
        company=company,
        location=location,
        match_summary=summary,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        matched_skills=matched_skills,
        skill_gaps=skill_gaps,
        preferred_skill_gaps=preferred_skill_gaps,
        required_skill_match_count=required_skill_match_count,
        required_skill_count=required_skill_count,
        required_skill_coverage=required_skill_coverage,
        candidate_evidence=candidate_evidence,
        job_evidence=job_evidence,
        recommendation=recommendation,
        provider="ollama" if llm_used else "deterministic_fallback",
        model=os.getenv("OLLAMA_MODEL", "qwen3:4b") if llm_used else "none",
        used_fallback=not llm_used,
    )
