import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "JOB_MATCH_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


st.set_page_config(
    page_title="AI Job-Matching Agent",
    page_icon="🎯",
    layout="wide",
)


class APIRequestError(Exception):
    """Raised when the FastAPI backend returns an error."""


def call_api(
    endpoint,
    file_bytes,
    filename,
    target_roles,
    top_k=None,
    job_id=None,
):
    """Send a PDF resume and form data to FastAPI."""
    data = {}

    if top_k is not None:
        data["top_k"] = str(top_k)

    if job_id is not None:
        data["job_id"] = str(job_id)

    files = [
        (
            "file",
            (filename, file_bytes, "application/pdf"),
        )
    ]

    files.extend(
        ("target_roles", (None, role))
        for role in target_roles
    )

    try:
        response = httpx.post(
            f"{API_BASE_URL}{endpoint}",
            files=files,
            data=data,
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
    except httpx.RequestError as exc:
        raise APIRequestError(
            f"Cannot connect to the backend at {API_BASE_URL}."
        ) from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}

    if not response.is_success:
        detail = payload.get("detail", "Unknown backend error.")
        raise APIRequestError(
            f"Backend error ({response.status_code}): {detail}"
        )

    return payload


def render_job_cards(
    jobs,
    method,
    resume_bytes=None,
    resume_filename=None,
    target_roles=None,
    enable_ai_analysis=False,
):
    """Render ranked job recommendations and optional AI analysis."""
    if not jobs:
        st.info("No recommendations were returned.")
        return

    score_fields = {
        "TF-IDF": "tfidf_similarity",
        "Semantic": "semantic_similarity",
        "Diversified": "semantic_similarity",
    }

    rank_fields = {
        "TF-IDF": "tfidf_rank",
        "Semantic": "semantic_rank",
        "Diversified": "diversified_rank",
    }

    score_field = score_fields[method]
    rank_field = rank_fields[method]

    for position, job in enumerate(jobs, start=1):
        rank = job.get(rank_field, position)
        title = job.get("title") or "Untitled position"
        company = job.get("company") or "Company not specified"
        location = job.get("location") or "Location not specified"
        job_id = job.get("job_id") or job.get("id")

        with st.container(border=True):
            st.write(f"**#{rank} — {title}**")
            st.caption(f"{company} | {location}")

            col1, col2, col3 = st.columns(3)

            with col1:
                score = job.get(score_field)
                if score is not None:
                    st.metric("Similarity", f"{float(score):.4f}")

            with col2:
                st.write("**Seniority**")
                st.write(job.get("seniority_level") or "Not specified")

            with col3:
                st.write("**Employment**")
                st.write(job.get("employment_type") or "Not specified")

            duplicate_count = job.get("duplicate_posting_count")

            if duplicate_count and duplicate_count > 1:
                st.caption(
                    f"Consolidated {duplicate_count} similar postings"
                )

            alternate_ids = job.get("alternate_job_ids")
            if alternate_ids:
                with st.expander("View alternate posting IDs"):
                    st.text(alternate_ids)

            if job_id:
                st.caption(f"Job ID: {job_id}")

            # AI analysis is enabled only in diversified recommendations.
            if (
                enable_ai_analysis
                and resume_bytes
                and resume_filename
                and job_id
            ):
                button_key = (
                    f"analyze_{method.lower().replace(' ', '_')}_{job_id}"
                )

                if st.button(
                    "Analyze this job with local AI",
                    key=button_key,
                    help="Analyze this posting against your uploaded resume.",
                ):
                    try:
                        with st.spinner(
                            "Qwen3 is analyzing this job locally..."
                        ):
                            analysis = call_api(
                                endpoint="/analyze-job",
                                file_bytes=resume_bytes,
                                filename=resume_filename,
                                target_roles=target_roles or [],
                                job_id=str(job_id),
                            )

                        st.session_state.setdefault(
                            "job_analyses", {}
                        )[str(job_id)] = analysis

                    except APIRequestError as exc:
                        st.error(str(exc))

                    except Exception as exc:
                        st.error("Unable to analyze this job.")
                        st.exception(exc)

                analysis = st.session_state.get(
                    "job_analyses", {}
                ).get(str(job_id))

                if analysis:
                    st.divider()
                    st.subheader("AI job analysis")

                    st.write(analysis["match_summary"])

                    coverage = analysis.get("required_skill_coverage")
                    matched_count = analysis.get(
                        "required_skill_match_count", 0
                    )
                    required_count = analysis.get("required_skill_count", 0)

                    if coverage is not None:
                        st.metric(
                            "Required-skill coverage",
                            f"{coverage:.0%}",
                            help=(
                                "Measures detected overlap in extracted "
                                "required skills. It is not an overall "
                                "job-fit score or hiring probability."
                            ),
                        )
                        st.progress(float(coverage))
                        st.caption(
                            f"{matched_count} of {required_count} "
                            "recognized required skills matched."
                        )
                    else:
                        st.info(
                            "Required-skill coverage is unavailable because "
                            "no clear required-skill list was extracted."
                        )

                    required_col1, required_col2 = st.columns(2)

                    with required_col1:
                        st.write("**Matched required skills**")
                        st.write(
                            ", ".join(analysis["matched_skills"])
                            or "No recognized overlap detected."
                        )

                    with required_col2:
                        st.write("**Missing required skills**")
                        st.write(
                            ", ".join(analysis["skill_gaps"])
                            or "No recognized required-skill gaps detected."
                        )

                    st.write("**Preferred skills**")
                    st.write(
                        ", ".join(analysis["preferred_skills"])
                        or "No preferred skills explicitly identified."
                    )

                    st.write("**Preferred skills not detected**")
                    st.write(
                        ", ".join(analysis["preferred_skill_gaps"])
                        or "No preferred-skill gaps detected."
                    )

                    st.write("**Recommendation**")
                    st.write(analysis["recommendation"])

                    with st.expander("Review supporting evidence"):
                        st.markdown("**Candidate evidence**")
                        candidate_evidence = analysis.get(
                            "candidate_evidence", []
                        )
                        if candidate_evidence:
                            for item in candidate_evidence:
                                st.text(item)
                        else:
                            st.caption(
                                "No candidate evidence records were returned."
                            )

                        st.markdown("**Job description evidence**")
                        job_evidence = analysis.get("job_evidence", [])
                        if job_evidence:
                            for item in job_evidence:
                                st.text(item)
                        else:
                            st.caption(
                                "No job-description evidence records were returned."
                            )

                    if analysis["used_fallback"]:
                        st.warning(
                            "Ollama was unavailable or its response failed "
                            "validation. This analysis uses the deterministic "
                            "Python fallback."
                        )
                    else:
                        st.caption(
                            f"Generated locally using {analysis['provider']} "
                            f"({analysis['model']})."
                        )

st.title("🎯 AI Job-Matching Agent")
st.write(
    "Upload your resume to discover job recommendations "
    "using lexical and semantic matching."
)

st.caption(f"Backend API: {API_BASE_URL}")

st.info(
    "Dataset notice: recommendations use an archived job corpus "
    "scraped on April 30–May 1, 2026. These are not verified live vacancies."
)

with st.form("resume_matching_form"):
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
        max_upload_size=MAX_UPLOAD_MB,
    )

    target_role_input = st.text_input(
        "Target roles",
        placeholder="Junior Data Analyst, Junior Data Scientist",
        help="Enter one or more roles separated by commas.",
    )

    top_k = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    submitted = st.form_submit_button(
        "Analyze resume and find matching jobs",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if uploaded_file is None:
        st.error("Please upload a PDF resume.")
        st.stop()

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.error("The uploaded file is empty.")
        st.stop()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("The resume exceeds the 10 MB file-size limit.")
        st.stop()

    target_roles = [
        role.strip()
        for role in target_role_input.split(",")
        if role.strip()
    ]

    try:
        with st.spinner("Extracting resume and candidate information..."):
            profile_data = call_api(
                endpoint="/upload-resume",
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                target_roles=target_roles,
            )

        with st.spinner(
            "Running TF-IDF and semantic job matching..."
        ):
            match_data = call_api(
                endpoint="/match",
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                target_roles=target_roles,
                top_k=top_k,
            )

        st.session_state["profile_data"] = profile_data
        st.session_state["match_data"] = match_data

        # Keep the uploaded PDF in this Streamlit session only.
        # It is not persisted to disk by this application.
        st.session_state["resume_bytes"] = file_bytes
        st.session_state["resume_filename"] = uploaded_file.name
        st.session_state["target_roles"] = target_roles
        st.session_state["job_analyses"] = {}

    except APIRequestError as exc:
        st.error(str(exc))

    except Exception as exc:
        st.error(
            "An unexpected error occurred while processing the resume."
        )
        st.exception(exc)


profile_data = st.session_state.get("profile_data")
match_data = st.session_state.get("match_data")

if profile_data and match_data:
    st.divider()
    st.header("Candidate profile")

    col1, col2, col3 = st.columns(3)

    col1.metric("Resume pages", profile_data["page_count"])
    col2.metric(
        "Extracted characters",
        profile_data["extracted_characters"],
    )
    col3.metric(
        "Detected skills",
        len(profile_data["skills"]),
    )

    with st.expander("Review extracted candidate information", expanded=True):
        st.write("**Detected technical skills**")
        st.write(", ".join(profile_data["skills"]) or "No skills detected.")

        st.write("**Education evidence**")
        for item in profile_data["education_evidence"]:
            st.write(f"- {item}")

        st.write("**Target roles**")
        st.write(
            ", ".join(profile_data["target_roles"])
            or "No target roles specified."
        )

        st.write("**Recognized resume sections**")
        st.write(", ".join(profile_data["sections"].keys()))

    st.divider()
    st.header("Job recommendations")

    eligible_count = match_data.get("eligible_job_count")
    excluded_count = match_data.get("excluded_job_count")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Eligible jobs",
            eligible_count if eligible_count is not None else "Unavailable",
        )

    with metric_col2:
        st.metric(
            "Excluded jobs",
            excluded_count if excluded_count is not None else "Unavailable",
        )

    st.caption(
        "Eligibility filters consider target role, location, seniority, "
        "experience requirements and applicable education evidence."
    )

    st.caption(
        f"Model device: {match_data['model_device']} | "
        f"Candidate chunks: {match_data['candidate_chunks']} | "
        f"Indexed job chunks: {match_data['indexed_chunks']}"
    )

    st.caption(
        "Similarity scores indicate relative text similarity. "
        "They are not probabilities of getting hired."
    )

    results = match_data["results"]

    tabs = st.tabs([
        "Diversified recommendations",
        "Semantic ranking",
        "TF-IDF baseline",
    ])

    with tabs[0]:
        render_job_cards(
            results["diversified"],
            "Diversified",
            resume_bytes=st.session_state.get("resume_bytes"),
            resume_filename=st.session_state.get("resume_filename"),
            target_roles=st.session_state.get("target_roles", []),
            enable_ai_analysis=True,
        )

    with tabs[1]:
        render_job_cards(
            results["semantic"],
            "Semantic",
        )

    with tabs[2]:
        render_job_cards(
            results["tfidf"],
            "TF-IDF",
        )

    st.success("Job matching completed successfully.")

st.divider()
st.caption(
    "AI Job-Matching Agent | Python · FastAPI · Streamlit · "
    "Sentence Transformers · FAISS · Scikit-learn"
)
