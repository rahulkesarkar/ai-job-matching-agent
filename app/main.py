import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.services.job_eligibility import (
    JobEligibilityError,
    filter_jobs_for_candidate,
)
from app.services.llm_job_analyzer import (
    JobAnalysisResult,
    analyze_job,
)
from starlette.concurrency import run_in_threadpool


from app.services.candidate_profile import CandidateProfileError
from app.services.job_matching_engine import (
    JobMatchingEngine,
    JobMatchingError,
)
from app.services.resume_parser import ResumeParsingError
from app.services.resume_pipeline import process_resume_pdf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class HealthResponse(BaseModel):
    status: str
    matching_engine_ready: bool


class ResumeProfileResponse(BaseModel):
    filename: str
    page_count: int
    extracted_characters: int
    skills: list[str]
    education_evidence: list[str]
    sections: dict[str, str]
    target_roles: list[str]
    profile_ready: bool


class MatchResponse(BaseModel):
    candidate_skills: list[str]
    target_roles: list[str]
    candidate_pages: int
    candidate_chunks: int
    indexed_chunks: int
    model_device: str
    eligible_job_count: int = 0
    excluded_job_count: int = 0
    results: dict[str, list[dict]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize expensive ML resources once at application startup.
    Reuse the same engine for subsequent requests.
    """
    logger.info("Starting AI Job-Matching Agent...")

    app.state.matching_engine = None

    try:
        app.state.matching_engine = JobMatchingEngine()
        logger.info("Matching engine initialized successfully.")
    except Exception:
        logger.exception("Matching engine initialization failed.")

    yield

    logger.info("Shutting down AI Job-Matching Agent...")
    app.state.matching_engine = None


app = FastAPI(
    title="AI Job-Matching Agent API",
    description=(
        "Resume ingestion, candidate profiling and "
        "AI-powered job recommendation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


async def read_resume_upload(file: UploadFile) -> bytes:
    """Read an uploaded PDF with a strict size limit."""
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Resume exceeds the 10 MB upload limit.",
        )

    return contents


def get_matching_engine(request: Request) -> JobMatchingEngine:
    """Return the initialized matching engine."""
    engine = getattr(request.app.state, "matching_engine", None)

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Matching engine is not currently available.",
        )

    return engine


def dataframe_records(dataframe, limit: int) -> list[dict]:
    """Convert ranked job results into JSON-safe records."""
    records = dataframe.head(limit).copy()

    records = records.astype(object).where(
        records.notna(),
        None,
    )

    return records.to_dict(orient="records")


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Report whether the application and matching engine are ready."""
    engine_ready = (
        getattr(request.app.state, "matching_engine", None)
        is not None
    )

    return HealthResponse(
        status="ready" if engine_ready else "degraded",
        matching_engine_ready=engine_ready,
    )


@app.post("/upload-resume", response_model=ResumeProfileResponse)
async def upload_resume(
    file: Annotated[UploadFile, File()],
    target_roles: Annotated[list[str] | None, Form()] = None,
):
    """
    Extract resume text and return a structured candidate profile.

    Resume content is processed in memory and is not persisted.
    """
    file_bytes = await read_resume_upload(file)

    try:
        result = process_resume_pdf(
            file_bytes=file_bytes,
            target_roles=target_roles,
        )
    except (ResumeParsingError, CandidateProfileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = result.profile

    return ResumeProfileResponse(
        filename=file.filename or "resume.pdf",
        page_count=result.extraction.page_count,
        extracted_characters=result.extraction.extracted_characters,
        skills=profile.skills,
        education_evidence=profile.education_evidence,
        sections=profile.sections,
        target_roles=profile.target_roles,
        profile_ready=bool(profile.matching_text.strip()),
    )


@app.post("/match", response_model=MatchResponse)
async def match_jobs(
    request: Request,
    file: Annotated[UploadFile, File()],
    target_roles: Annotated[list[str] | None, Form()] = None,
    top_k: Annotated[int, Form(ge=1, le=50)] = 10,
):
    """
    Process a resume and generate three ranked recommendation lists.

    The endpoint performs PDF extraction, candidate profiling,
    TF-IDF retrieval, semantic retrieval and diversification.
    """
    engine = get_matching_engine(request)
    file_bytes = await read_resume_upload(file)

    try:
        ingestion = process_resume_pdf(
            file_bytes=file_bytes,
            target_roles=target_roles,
        )

        results = engine.match(ingestion.profile)

    except (ResumeParsingError, CandidateProfileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except JobMatchingError as exc:
        logger.exception("Job matching failed.")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate job recommendations.",
        ) from exc

    return MatchResponse(
        candidate_skills=ingestion.profile.skills,
        target_roles=ingestion.profile.target_roles,
        candidate_pages=ingestion.extraction.page_count,
        candidate_chunks=results.candidate_chunk_count,
        indexed_chunks=results.indexed_chunk_count,
        model_device=results.model_device,
        eligible_job_count=getattr(results, "eligible_job_count", 0),
        excluded_job_count=getattr(results, "excluded_job_count", 0),
        results={
            "tfidf": dataframe_records(results.tfidf_ranked, top_k),
            "semantic": dataframe_records(results.semantic_ranked, top_k),
            "diversified": dataframe_records(
                results.diversified_ranked,
                top_k,
            ),
        },
    )


@app.post("/analyze-job", response_model=JobAnalysisResult)
async def analyze_selected_job(
    request: Request,
    file: Annotated[UploadFile, File()],
    job_id: Annotated[str, Form(min_length=1)],
    target_roles: Annotated[list[str] | None, Form()] = None,
):
    """
    Generate an evidence-grounded explanation for one eligible job.

    The uploaded resume is processed in memory. The selected job ID is
    checked against the candidate-specific eligibility rules before
    the local LLM analyzer is called.
    """
    engine = get_matching_engine(request)
    file_bytes = await read_resume_upload(file)

    try:
        ingestion = process_resume_pdf(
            file_bytes=file_bytes,
            target_roles=target_roles,
        )

        eligibility = filter_jobs_for_candidate(
            jobs=engine.eligible_jobs,
            target_roles=ingestion.profile.target_roles,
            max_required_experience_years=1,
            preferred_country="India",
            candidate_profile=ingestion.profile,
        )

    except (ResumeParsingError, CandidateProfileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except JobEligibilityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    eligible_jobs = eligibility.eligible_jobs

    selected = eligible_jobs.loc[
        eligible_jobs["id"].astype(str).eq(job_id)
    ]

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail="The selected job is not eligible for this candidate.",
        )

    job_record = selected.iloc[0].to_dict()

    try:
        analysis = await run_in_threadpool(
            analyze_job,
            ingestion.profile,
            job_record,
        )
    except (TypeError, ValueError) as exc:
        logger.exception("Selected job analysis failed.")
        raise HTTPException(
            status_code=422,
            detail="The selected job could not be analyzed.",
        ) from exc

    logger.info(
        "Job analysis completed: job_id=%s provider=%s fallback=%s",
        job_id,
        analysis.provider,
        analysis.used_fallback,
    )

    return analysis
