from dataclasses import dataclass
from typing import Sequence

from app.services.candidate_profile import (
    CandidateProfile,
    build_candidate_profile,
)
from app.services.resume_parser import (
    ResumeExtraction,
    extract_resume_text,
)


@dataclass
class ResumeIngestionResult:
    extraction: ResumeExtraction
    profile: CandidateProfile


def process_resume_pdf(
    file_bytes: bytes,
    target_roles: Sequence[str] | None = None,
) -> ResumeIngestionResult:
    """
    Process an uploaded PDF resume into a structured candidate profile.

    Pipeline:
    1. Validate and extract PDF text.
    2. Identify skills, education and resume sections.
    3. Add explicitly supplied target roles.
    4. Return extraction metadata and the candidate profile.
    """
    extraction = extract_resume_text(file_bytes)

    profile = build_candidate_profile(
        resume_text=extraction.text,
        target_roles=target_roles,
    )

    return ResumeIngestionResult(
        extraction=extraction,
        profile=profile,
    )
