# Engineering Notes — AI Job-Matching Agent

This file records the implemented system, verified historical evidence, open engineering risks and the next validation work. Update it when a meaningful milestone is completed; do not mark work complete based only on source-code inspection.

## 1. Development environment

The development machine used for this project is a MacBook Air with Apple Silicon (ARM64), macOS 26.3, and Python 3.14.3. The project virtual environment is `.venv314`.

Use the project interpreter explicitly when running local checks:

```bash
./.venv314/bin/python -m pytest -v
```

The environment and hardware above describe the development setup, not a verified production deployment target.

## 2. Implemented application

The repository currently contains the following application components:

- PDF resume extraction with PyMuPDF and structured candidate-profile extraction.
- Explicit skill and education evidence extraction.
- User-selected target roles and candidate-aware eligibility heuristics for role, location, experience, seniority and education.
- TF-IDF lexical retrieval using scikit-learn.
- Semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).
- Token-aware job chunking (200-token chunks, 40-token overlap).
- FAISS `IndexFlatIP` over normalized vectors; normalized inner product is cosine-equivalent.
- Job-level maximum-score aggregation and near-duplicate recommendation diversification (current similarity threshold 0.995).
- Local Ollama integration with `qwen3:4b` for selected-job analysis.
- Python-owned skill-match, coverage and gap calculations; Pydantic validation and unsupported-claim checks; deterministic fallback.
- FastAPI endpoints: `GET /health`, `POST /upload-resume`, `POST /match`, and `POST /analyze-job`.
- Streamlit user interface for resume upload, target-role selection, recommendation comparison and selected-job analysis.

The implementation is a local prototype, not a production hiring system. Eligibility rules are heuristics; absence of an extracted skill is not proof that a candidate lacks it. Skill coverage is not a probability of interview or employment.

## 3. Data pipeline and corpus

Source: [Role Radar dataset](https://huggingface.co/datasets/oksomu/role-radar-dataset).

Recorded processing counts:

| Stage | Records |
|---|---:|
| Source records | 2,500 |
| Cleaned unique job records | 2,471 |
| Semantic-matching-eligible jobs | 2,468 |
| Generated semantic chunks | 10,095 |

The corpus was scraped on April 30 and May 1, 2026. It is archived data and must not be presented as a live or currently open vacancy feed.

Raw data, processed datasets, embeddings and FAISS artifacts are excluded from Git and must be provisioned separately for a local or hosted runtime. The dataset card advertises Apache-2.0, but redistribution rights for individual third-party job descriptions have not been independently established. Do not commit or redistribute the raw corpus without resolving those rights.

The pipeline includes location normalization, duplicate cleaning, description-quality checks, token auditing, chunk generation, embedding generation and FAISS indexing. The reproducible entry point is `scripts/prepare_data.py`; run without `--run` to validate inputs and with `--run` to generate artifacts.

## 4. Retrieval evaluation — preliminary evidence

The recorded manual evaluation covered 31 reviewed postings, grouped into 27 recommendation groups, with 7 relevant groups. The recorded metrics are:

| Metric | TF-IDF | Semantic | Diversified semantic |
|---|---:|---:|---:|
| Precision@5 | 0.6000 | 0.6000 | 0.6000 |
| Precision@10 | 0.4000 | 0.4000 | 0.4000 |
| Recall@10 within judged groups | 0.5714 | 0.5714 | 0.5714 |
| MRR | 1.0000 | 1.0000 | 1.0000 |
| NDCG@10 | 0.7316 | 0.7657 | 0.7657 |

These are historical, preliminary results from one synthetic candidate profile and a small judged pool. Recall is calculated only within judged groups, not across the entire corpus. Semantic retrieval improved NDCG@10 in this pool, but did not improve recorded precision or recall. Do not generalize these results to all candidates or job families.

**Next evaluation work:** add independently judged examples across multiple candidate profiles, role families and experience levels; preserve the judgment protocol; recalculate metrics and document sample sizes and uncertainty.

## 5. Test evidence and verification status

A previous full test run recorded **92 passed, 5 warnings**. The warnings were reported as non-blocking SWIG/PyMuPDF deprecation warnings. This is historical evidence only; it is not proof that the current repository head passes.

The current completion pass must rerun:
- Full pytest suite.
- API integration smoke test (`scripts/smoke_test_api.py`).
- Local FastAPI health and endpoint checks.
- Manual Streamlit workflow, including actual Ollama analysis and deterministic fallback.
- Synthetic PDF generation and PDF content/page-count verification.
- Fresh-environment setup and data-artifact provisioning where feasible.

No new local test run, synthetic PDF generation, screenshot capture or screen recording is claimed by this documentation update. These actions require the user's Mac and local runtime.

## 6. Security, reliability and production gaps

The local prototype is not yet approved for unrestricted public uploads. Before exposing it to recruiters, assess and implement at minimum:

- Authentication or an appropriate abuse-control boundary for public endpoints.
- Rate limiting and concurrency limits, especially for `/analyze-job`.
- Request, upload, page-count, parsing-time and inference-time limits.
- Production CORS allowlist and secure reverse-proxy/TLS configuration.
- Safe error responses and operational logs that never include resume text or extracted personal data.
- Secret and environment-variable handling.
- Health/readiness checks that distinguish application startup from model/index readiness.
- Resource, latency and failure-recovery tests on the actual hosting target.

The parser currently documents a 10 MB upload limit and a 20-page limit. The resume parser processes content in memory rather than persisting the uploaded PDF. These controls do not by themselves make a public upload endpoint production-safe.

## 7. Deployment decision and constraints

The application has **not** been deployed as a publicly usable AI service. The portfolio website being published does not mean the AI application is deployed.

The user has asked for deployment, but the exact public-use requirement and acceptable hosting trade-offs still need confirmation. Do not silently substitute a recorded local demo for a requested online application, or promise always-on free inference.

Before choosing a host, validate the complete dependency stack on the target OS/architecture: Python 3.14 wheels, PyTorch, Sentence Transformers, FAISS, memory, disk, model download and inference. The job corpus and generated artifacts also need a lawful, reproducible provisioning strategy. Ollama/Qwen3 inference is a separate deployment constraint from hosting FastAPI and Streamlit.

## 8. Recruiter demonstration package

The repository includes:
- `data/sample/demo_candidate_resume.txt`: fictional candidate data using the reserved `.invalid` email domain.
- `scripts/create_demo_resume.py`: creates and validates a one-page A4 PDF.
- `docs/demo-recording-guide.md`: local setup, acceptance workflow, screenshot plan and recording outline.

The generated PDF has not yet been executed and verified on the user's Mac in this milestone. Genuine screenshots and recordings must be captured from the running application; never fabricate or recreate application evidence.

Planned screenshots:
- `01-upload-screen.png`
- `02-candidate-profile.png`
- `03-recommendations.png`
- `04-ai-job-analysis.png`

The YouTube presentation is a later deliverable. First stabilize the application and capture genuine evidence for the portfolio.

## 9. Portfolio and remaining user-dependent tasks

The portfolio repository is separate: [rahulkesarkar.github.io](https://github.com/rahulkesarkar/rahulkesarkar.github.io). Its project case-study page already describes the system and local-demo approach. Update it with real screenshots, a demo and a deployment link only after those assets and services exist and have been tested.

The homepage resume-download section is intentionally pending until the user confirms the correct, current PDF. Do not publish older resume versions or unconfirmed education details.

## 10. Next engineering sequence

1. Fresh local regression and integration verification.
2. Fix issues found and repeat the checks.
3. Expand relevance evaluation with multiple candidate profiles.
4. Confirm deployment target and public-use expectations; complete security and runtime validation.
5. Deploy and independently test the real application if the chosen host supports the full stack.
6. Capture genuine screenshots and recordings from the verified runtime.
7. Update README, walkthrough and portfolio with verified evidence.
8. Obtain the approved resume PDF, add direct-download behavior, and complete recruiter-journey QA.
