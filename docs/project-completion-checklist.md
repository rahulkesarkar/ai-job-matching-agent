# Project Completion Checklist

This checklist reflects the agreed portfolio demonstration scope. The application is intended to run locally for a genuine recorded demonstration; it is **not** intended to be an always-on public resume-upload service. Status must reflect evidence, not intention.

## A. Repository and CI work

- [x] Add the Python 3.14 GitHub Actions regression workflow.
- [x] Confirm the GitHub Actions regression workflow succeeds on `main` (November 24, 2026; commit `9ecdbfe6bf3885bbc4a379bee36ca77264c3e1a0`).
- [ ] Align remaining documentation and portfolio wording with the local-demo-only scope.
- [ ] Add genuine screenshots under `docs/demo/screenshots/`.
- [ ] Add a compressed real screen recording under `docs/demo/` or link a GitHub-hosted video asset.
- [ ] Finalize documentation after local verification and expanded evaluation.

## B. Requires the owner's development Mac

- [x] Align the Streamlit file-uploader limit with the 10 MB application limit in source code (local browser confirmation pending).
- [ ] Confirm the displayed 10 MB widget limit and readable supporting-evidence records in the updated local Streamlit UI.

- [ ] Pull the latest `main` branch.
- [ ] Generate `data/sample/demo_candidate_resume.pdf` and inspect that it is a complete one-page A4 PDF.
- [x] Run the full test suite with `.venv314` (November 24, 2026: 92 passed, 5 non-blocking warnings).
- [x] Run `scripts/smoke_test_api.py` (November 24, 2026: passed).
- [x] Confirm the local dataset, embeddings and FAISS index are present and load correctly (2,468 jobs, 10,095 chunks, 384 dimensions).
- [x] Start/verify FastAPI and `/health` (HTTP 200, matching engine ready).
- [x] Start Streamlit and complete an actual local PDF-upload workflow (private resume; no public use of its data).
- [x] Verify candidate profile extraction, eligibility counts and target-role filtering (role-family audit recorded in engineering notes).
- [x] Inspect TF-IDF, semantic and diversified recommendations (top-five comparison recorded in engineering notes).
- [x] Run Qwen3 4B selected-job analysis through local Ollama (provider confirmed as `ollama`).
- [x] Verify deterministic fallback when Ollama is unavailable in the live UI; invalid-response fallback is covered by automated tests.
- [ ] Restart and repeat the workflow; record errors, latency and resource usage.
- [ ] Capture the four genuine application screenshots listed in the demo guide.
- [ ] Record and review the actual local application walkthrough.
- [ ] Keep personal resumes, private information and unrelated desktop content out of captures.

## C. Evaluation and technical quality

- [ ] Expand relevance judgments beyond the current 31 reviewed postings and one synthetic candidate profile.
- [ ] Evaluate multiple candidate profiles, role families and experience levels.
- [ ] Recalculate Precision@5, Precision@10, Recall@10 (with scope stated), MRR and NDCG@10.
- [x] Inspect Data Analyst eligibility-filter exclusions for the local demonstration candidate (21 title matches, 5 eligible, 16 excluded; detailed scope and overlapping reasons documented).
- [ ] Document the limits of the archived dataset and heuristic rules.
- [ ] Review setup instructions against a fresh local environment.

## D. GitHub and recruiter-facing presentation

- [ ] Add real screenshots and concise captions to the repository.
- [ ] Add the real screen-recording video or a durable GitHub-hosted video link.
- [ ] Make the README lead recruiters to the screenshot gallery and video.
- [ ] Verify assets and links from a signed-out browser session.
- [ ] Update the portfolio case study with real screenshot previews and video/repository links.
- [ ] Test the recruiter journey: portfolio homepage → project case study → GitHub → screenshots/video/source.
- [ ] Complete desktop and mobile visual/accessibility checks.

## E. Requires owner approval or assets

- [ ] Confirm the correct current resume PDF before adding the homepage download button.
- [ ] Review and approve the genuine screenshots and recording.
- [ ] Approve the final recruiter-facing portfolio presentation.
- [ ] Prepare and publish the YouTube presentation as a later deliverable.

## Explicitly out of scope

- Public resume uploads by recruiters.
- A continuously running public backend or frontend.
- 24/7 hosting, uptime commitments, public inference and hosting-cost optimization.
- Production authentication, public API rate limiting and public-service load testing.

Keep the prototype bound to localhost for the recording. Do not expose upload or inference endpoints to the public internet. Revisit production controls only if the product scope changes to a public interactive service.

## Completion gate

The project is complete for the agreed portfolio scope when the current local application passes fresh verification, the demonstration is recorded from a real run, genuine screenshots and video are available from GitHub, evaluation and limitations are accurately documented, and the portfolio-to-repository recruiter journey has been tested. Public deployment is not a completion requirement.