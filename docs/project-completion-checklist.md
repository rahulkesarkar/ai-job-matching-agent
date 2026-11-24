# Project Completion Checklist

This checklist separates work that can be advanced in the repository from checks that require the owner's Mac, assets or decisions. Status must reflect evidence, not intention.

## A. Remote/repository work

- [x] Review current project documentation against the implemented architecture.
- [x] Replace obsolete engineering notes that described implemented features as not yet started.
- [x] Label the previously recorded 92-test result as historical rather than current verification.
- [x] Clarify that the AI application is not publicly deployed.
- [x] Document the deployment decision and production-readiness constraints.
- [x] Add a reproducible recruiter-demo acceptance and capture guide.
- [x] Prevent the synthetic PDF generator from saving an overflowing one-page document.
- [ ] Review all setup instructions against a fresh local installation.
- [ ] Add or verify automated CI for supported Python/dependency combinations.
- [ ] Reassess dependency pins and platform compatibility for the selected deployment target.
- [ ] Expand evaluation methodology and judgments across multiple candidate profiles.
- [ ] Complete production security review and implement the required controls.
- [ ] Deploy the real application and verify the hosted user journey.

## B. Requires the owner's development Mac

- [ ] Pull the latest repository branch/merge.
- [ ] Generate and inspect `data/sample/demo_candidate_resume.pdf`.
- [ ] Run the full test suite and retain the actual output.
- [ ] Run `scripts/smoke_test_api.py`.
- [ ] Verify FastAPI readiness and Streamlit startup with the prepared local data/index.
- [ ] Complete manual matching and Qwen3 analysis.
- [ ] Verify deterministic fallback when Ollama is unavailable or returns invalid output.
- [ ] Record errors, latency and resource usage; fix and rerun failures.
- [ ] Capture genuine screenshots and screen recordings from the running app.

## C. Requires owner confirmation or approved assets

- [ ] Confirm whether recruiters must be able to upload resumes and use the application online, or whether the public portfolio should link to a restricted/demo deployment.
- [ ] Confirm acceptable hosting constraints, including cost, uptime, cold starts and inference latency.
- [ ] Provide/approve the correct current resume PDF before enabling a homepage download.
- [ ] Review the real screenshots and recording before publication.
- [ ] Approve the final recruiter-facing portfolio presentation.
- [ ] Prepare and publish the YouTube presentation later.

## Completion gate

Do not mark the project complete until the real application has passed fresh local tests, the chosen deployment mode has been validated, genuine demo evidence is published, and the portfolio-to-project-to-source journey has been checked. Never substitute a recorded local demo for a requested public application deployment.
