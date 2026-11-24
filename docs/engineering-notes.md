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

## 6. Security, reliability and demonstration boundary

The agreed portfolio scope is a local-only application demonstration. Recruiters will inspect the source, genuine screenshots and a recording of the application running locally; they will not upload their own resumes or use a public inference endpoint.

Keep FastAPI and Streamlit bound to localhost during development and recording. Do not expose the current upload or LLM-analysis endpoints to the public internet. Authentication, public API rate limiting, public-service load testing and production TLS/CORS configuration are not completion requirements for this local-only scope.

The parser currently documents a 10 MB upload limit and a 20-page limit. Resume content is processed in memory rather than persisted by the resume parser. These protections do not make the current code suitable for unrestricted public exposure. If the product scope changes to a public interactive service, complete a separate security review covering authentication/abuse controls, request and resource limits, safe logging, CORS, TLS, readiness, concurrency and failure recovery.

## 7. Deployment decision and constraints

**Decision: public deployment is explicitly out of scope for the agreed portfolio demonstration.** The user does not want an always-on service that accepts recruiter resume uploads or incurs continuous hosting/inference costs.

The intended proof is a reproducible local run captured in a genuine screen recording, supported by genuine screenshots and the public GitHub source. No hosted application URL, uptime claim or public inference endpoint should be added to the portfolio.

The local prototype still needs fresh Mac verification. The Apple Silicon development environment and local Ollama/Qwen3 runtime are the target for the demonstration, not a production hosting platform. Any future change to public access would require a new architecture and security decision.

## 8. Recruiter demonstration package

The repository includes:
- `data/sample/demo_candidate_resume.txt`: fictional candidate data using the reserved `.invalid` email domain.
- `scripts/create_demo_resume.py`: creates and validates a one-page A4 PDF.
- `docs/demo-recording-guide.md`: local setup, acceptance workflow, screenshot plan, recording outline and GitHub evidence package.

The generated PDF has not yet been executed and verified on the user's Mac. Genuine screenshots and a recording must be captured from the running local application; never fabricate or recreate application evidence.

Planned screenshot files:
- `docs/demo/screenshots/01-upload-screen.png`
- `docs/demo/screenshots/02-candidate-profile.png`
- `docs/demo/screenshots/03-recommendations.png`
- `docs/demo/screenshots/04-ai-job-analysis.png`

The intended video is a compressed recording of the real local end-to-end workflow. The YouTube presentation is a separate later deliverable.

## 9. Portfolio and remaining user-dependent tasks

The portfolio repository is separate: [rahulkesarkar.github.io](https://github.com/rahulkesarkar/rahulkesarkar.github.io). Its project case-study page already describes the system and local-demo approach. Update it with real screenshots and the recorded local demo only after those assets exist and have been reviewed. Do not add a deployment link; public hosting is outside the agreed scope.

The homepage resume-download section is intentionally pending until the user confirms the correct, current PDF. Do not publish older resume versions or unconfirmed education details.

## 10. Next engineering sequence

1. Complete fresh local regression and API integration verification on the Mac.
2. Generate and inspect the synthetic one-page resume PDF.
3. Run the full Streamlit workflow with the local data/index and Ollama; verify deterministic fallback.
4. Fix issues found and repeat the checks.
5. Expand relevance evaluation with multiple candidate profiles and document actual results.
6. Capture genuine screenshots and a screen recording from the verified local application.
7. Add the screenshot gallery and video to the GitHub repository and README.
8. Update the portfolio case study with the real evidence and verify the recruiter journey.
9. Obtain the approved resume PDF, add direct-download behavior, and complete final portfolio QA.

Public deployment, 24/7 uptime and public resume uploads are not completion milestones for this agreed scope.


## 11. Fresh local verification milestone — November 24, 2026

The user ran the current local application on an Apple Silicon Mac using Python 3.14.3 and the `.venv314` environment. The following evidence was supplied from actual local execution:

- Full pytest suite: **92 passed, 0 failed, 5 non-blocking SWIG/PyMuPDF deprecation warnings**, 8.40 seconds.
- `scripts/smoke_test_api.py`: passed. FastAPI initialized the matching engine; `/health` returned ready; PDF extraction, upload, candidate skills, target-role propagation, real matching, eligibility counters and all three recommendation result lists passed.
- Runtime artifacts loaded: 2,468 semantic-eligible jobs, 10,095 indexed chunks, 384-dimensional embeddings; model device was `mps:0`.
- Local Ollama integration: `scripts/smoke_test_llm_analysis.py` completed successfully using `qwen3:4b`, with `provider=ollama` and no fallback.
- FastAPI `/health` returned HTTP 200 with `matching_engine_ready=true`. Streamlit's `/_stcore/health` endpoint also returned HTTP 200.
- Manual Streamlit run using a private resume: PDF upload and extraction completed; 11 skills were detected; the selected role was Junior Data Analyst; the UI displayed the three retrieval lists and selected-job analysis. Do not store or publish the private resume or its personal details.
- Manual retrieval comparison: the first five results were identical across TF-IDF, semantic and diversified semantic ranking for this candidate. The similarity scores are method-specific and must not be compared across methods as though they shared a scale.
- Manual local-LLM outage test: Ollama was stopped while the app remained running. Selected-job analysis completed using the disclosed deterministic fallback. Ollama was then restarted; a fresh analysis again reported `ollama (qwen3:4b)`.
- Eligibility audit for the private resume and Junior Data Analyst role: 5 eligible and 2,463 excluded from 2,468 source jobs. Exclusion-reason counts overlap because one job may violate multiple rules. A role-family sensitivity diagnostic returned:

| Target role family | Title matches | Eligible at 1 year | Eligible at 2 years | Eligible at 3 years |
|---|---:|---:|---:|---:|
| Data Analyst | 21 | 5 | 5 | 5 |
| Data Scientist | 27 | 5 | 5 | 7 |
| Machine Learning Engineer | 25 | 5 | 7 | 13 |
| All three families | 70 | 14 | 16 | 24 |

These counts are diagnostic, not relevance labels or evidence that every result is suitable. Do not loosen the default experience threshold merely to inflate result volume. Keep recruiter demo claims scoped to the explicitly selected roles and describe limited corpus coverage honestly.

- Evidence-generation diagnostic found 7 non-empty candidate evidence records and 4 non-empty job evidence records for the selected RegisterKaro posting. The actual text display in the Streamlit expander has not yet been visually confirmed; inspect it before final screenshots.
- The upload widget displayed Streamlit's 200 MB default, while the application has a 10 MB client-side limit. Align the displayed/configured limit before recording.
- The private resume's education extraction was not fully reconciled with the user's confirmed education details. Do not use its profile in public screenshots or recording. Use the approved synthetic demonstration resume for public evidence.

Still not verified: synthetic PDF generation and inspection; local API fallback from malformed model output in the live UI (unit tests cover invalid output); a full app restart-and-repeat test; final evidence rendering; four genuine screenshots; screen recording; repository/portfolio evidence integration. These remain open until actually performed.


## 12. Focused Data Analyst eligibility audit — November 24, 2026

A local diagnostic used the user's private resume with the explicitly selected target role `Junior Data Analyst`. The resume content was not published. Among the 2,468 semantic-eligible source jobs:

- 21 job titles matched the Data Analyst title category.
- 5 passed all candidate eligibility filters.
- 16 were excluded.
- 14 of the excluded title matches carried a senior/leadership classification under the current title/seniority heuristic.
- Other overlapping exclusion reasons included explicit experience requirements from 2 to 9 years. Some jobs violated both seniority and experience rules.
- The source rows in this title-matched excluded set were India-classified; country filtering was not the cause of these 16 exclusions.

The counts of exclusion reasons overlap and must not be summed as unique jobs. The current conservative filters were retained; there was no evidence from this single candidate audit to justify loosening the rules. The result exposes a dataset-coverage limitation for the Data Analyst title category and a dependency on potentially imperfect job-platform seniority metadata. The audit is specific to this resume, target role and archived corpus; it does not prove that all classifications are correct.


## 13. Streamlit UI consistency correction — November 24, 2026

The live UI showed a 200 MB file-uploader allowance while the application itself enforced a 10 MB limit. The Streamlit uploader now receives `max_upload_size=10`, and both the widget and byte-level validation derive from the shared `MAX_UPLOAD_MB = 10` constant. This prevents the UI from inviting files that the application subsequently rejects. The change uses the supported per-widget Streamlit parameter; local browser re-verification is still pending.

The supporting-evidence expander was also changed to render each candidate and job evidence record using plain-text blocks, with explicit empty-state messages. The local diagnostic found seven non-empty candidate evidence records and four non-empty job evidence records, but the revised presentation must still be checked in the running browser before screenshots are captured.


## 14. Browser confirmation of Streamlit UI corrections — November 24, 2026

After synchronizing local branch `local-demo-verification` to commit `2355c30`, the user refreshed the actual local Streamlit browser application and confirmed:
- The resume uploader now displays a 10 MB limit, consistent with the application's 10 MB byte validation.
- The supporting-evidence expander renders candidate resume evidence and job-description evidence as readable text records. The local diagnostic had counted 7 candidate records and 4 job-description records.

GitHub Actions for commit `2355c30fcaa02aad93fd655c976bf6503b9b5f36` completed successfully on Python 3.14 (regression run 45364348161). The latest code and UI corrections therefore have both successful CI coverage and local browser confirmation.

Still pending before recording: create and inspect the synthetic demo PDF; verify a fresh clean-session run with synthetic data; capture and review four genuine screenshots and one real screen recording; publish and integrate those assets into GitHub and the portfolio.


## 15. Synthetic-candidate end-to-end demonstration verification — November 24, 2026

The user completed a full manual workflow in the local Streamlit application using the corrected fictional `data/sample/demo_candidate_resume.pdf` for Aarav Mehta, with target roles `Junior Data Analyst` and `Data Analyst Intern`.

Observed results supplied from the running browser:
- PDF upload and extraction succeeded; one page and 1,712 extracted characters.
- 14 skills were detected, including Python, SQL, Pandas, NumPy, Scikit-learn, Data Analysis, Data Cleaning, Feature Engineering, Statistics, Data Visualization, Power BI, Excel, Git and GitHub.
- Fictional Bachelor of Science (B.Sc.) in Data Analytics education evidence was extracted.
- Five jobs passed eligibility filtering from the 2,468-job corpus; 2,463 were excluded under the documented role, country, seniority, experience and education heuristics.
- TF-IDF, semantic and diversified recommendation views completed. The top semantic result was Data Analyst Intern at RegisterKaro (similarity 0.6840); scores are method-specific and are not hiring probabilities.
- Selected-job analysis completed using local `ollama (qwen3:4b)`.
- Python-calculated required-skill coverage was 4/5 (80%): Excel, SQL, Power BI and Python matched; Tableau was not detected as required. Data Visualization was classified as preferred and detected in the candidate skills.
- Supporting evidence was displayed as readable text: candidate evidence (skills, retail analysis project, Power BI dashboard project and fictional education) and job-description evidence (SQL, BI dashboards, Excel reporting and Advanced Excel requirements).

This is a successful, manually observed synthetic-resume end-to-end run and is suitable as the source for the public demonstration once a final clean/restarted run and evidence capture are completed. It is not a claim that current vacancies were found; the corpus remains archived. The private resume must not be included in the public evidence package.

## 16. Seven-screenshot local evidence capture — November 24, 2026

Seven cropped JPG screenshots were captured from the local Streamlit demonstration using the fictional candidate:

1. Resume upload
2. Candidate profile
3. TF-IDF recommendations
4. Semantic recommendations
5. Diversified recommendations
6. AI job analysis
7. Supporting evidence

The final files are stored in `docs/demo/screenshots/` with numbered descriptive filenames. The earlier timestamp-named PNG copies were removed from the project screenshot directory; the Desktop originals were left untouched.

The README gallery, demo recording guide and completion checklist now reflect the seven-image set. The screenshots still require a final privacy and readability review before public release. The real screen recording and portfolio integration remain pending.

## 17. Local demo video published — November 24, 2026

The completed macOS screen recording was converted from MOV to MP4 using stream copy (no re-encoding), preserving the H.264 video and AAC audio streams. The published file is `docs/demo/ai-job-matching-agent-demo.mp4` (44,646,940 bytes; approximately 43 MiB; 3:21; 2880 × 1800). The original MOV remains on the user's Desktop.

The MP4 has been pushed to the public project repository. The README links to the recording, and the portfolio case study includes an HTML5 video player alongside the seven genuine screenshots. Final playback and audio checks on desktop and mobile remain to be completed.
