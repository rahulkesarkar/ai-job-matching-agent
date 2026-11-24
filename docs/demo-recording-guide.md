# Recruiter Demo Recording Guide

## Purpose and presentation rules

Create a short, factual demonstration of the real locally running AI Job-Matching Agent. The recording and screenshots must come from the actual Streamlit application after a fresh end-to-end check.

- Use only the synthetic candidate resume in `data/sample/demo_candidate_resume.pdf`.
- Do not upload or expose Rahul's personal resume or another real person's resume.
- Do not imply that archived job postings are currently open vacancies.
- Do not describe similarity or skill-coverage scores as hiring probabilities.
- Do not edit, fabricate, or manually alter application results in screenshots.
- Do not claim the app is publicly deployed unless the hosted application has completed a real end-to-end acceptance test. The local recording workflow remains useful while deployment is being validated.

## 1. Prepare the synthetic PDF

From the repository root, with the project virtual environment active:

```bash
python scripts/create_demo_resume.py
```

The script creates and re-opens a one-page PDF at:

`data/sample/demo_candidate_resume.pdf`

The candidate is fictional. The email uses the reserved `example.invalid` domain.

## 2. Verify the local environment

Confirm that the required prepared artifacts exist locally. Raw data, processed datasets, embeddings and the FAISS index are intentionally not committed to Git; follow the README's dataset preparation steps if they are absent.

Confirm Ollama is available and the configured model is installed:

```bash
ollama list
```

The default model is `qwen3:4b`. Start Ollama if it is not already running.

Run the recorded regression suite:

```bash
python -m pytest -v
```

Run the real FastAPI integration smoke test:

```bash
python scripts/smoke_test_api.py
```

The smoke test checks API startup, PDF upload, candidate extraction, actual matching, eligibility counters and result lists. It does not replace the separate manual Streamlit and LLM-analysis walkthrough.

## 3. Start the application

Load the local environment variables in each terminal session as described in the README.

Terminal 1 — FastAPI:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — Streamlit:

```bash
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Check `http://127.0.0.1:8000/health` and open `http://127.0.0.1:8501`.

## 4. Manual end-to-end acceptance checklist

1. Upload `data/sample/demo_candidate_resume.pdf`.
2. Enter `Junior Data Analyst` as the target role.
3. Submit the form and wait for processing to finish.
4. Inspect the extracted candidate profile and recognized skills.
5. Confirm the eligible and excluded job counts are visible and plausible.
6. Review TF-IDF, semantic and diversified recommendation tabs.
7. Inspect the returned job titles and metadata; do not imply archived jobs are currently open.
8. In the diversified recommendations, choose one returned job and select **Analyze this job with local AI**.
9. Wait for Qwen3 analysis; inspect matched skills, skill gaps, evidence and recommendation.
10. Confirm any fallback notice is visible if the local LLM is unavailable or returns invalid output.
11. Repeat once after a clean app restart to confirm the workflow is reproducible.

Record the actual outcome, including errors and latency. Do not mark the checklist complete until it has been performed on the local machine.

## 5. Screenshot capture plan

Capture unmodified screenshots from the running app:

- `01-upload-screen.png` — upload and target-role form.
- `02-candidate-profile.png` — extracted candidate evidence and eligibility counts.
- `03-recommendations.png` — recommendation tabs and actual ranked results.
- `04-ai-job-analysis.png` — selected-job analysis, skill matches, gaps and evidence.

Before capture, close unrelated windows, hide notifications and ensure no personal files or browser tabs are visible. Keep the app's real content and status messages intact.

## 6. Video outline (approximately 2–3 minutes)

- 0:00–0:15 — State the problem and what the prototype demonstrates.
- 0:15–0:40 — Upload the synthetic PDF and select the target role.
- 0:40–1:05 — Show the extracted candidate profile and eligibility filtering.
- 1:05–1:40 — Compare TF-IDF, semantic and diversified recommendations.
- 1:40–2:20 — Analyze one selected job and explain the evidence, matched skills and gaps.
- 2:20–2:40 — Close with the architecture and GitHub repository.

Use a 16:9 desktop capture, readable browser zoom, clean audio and no unnecessary terminal output. Do not speed through loading states or conceal failures. If analysis uses deterministic fallback, explain that honestly.

## 7. Portfolio integration acceptance

The case-study page should contain:

- A playable video hosted from a stable public video URL or a repository asset.
- A small gallery of the genuine screenshots.
- Captions explaining what each screen demonstrates.
- A visible GitHub source link.
- A clear note that this is a locally run demonstration using an archived job corpus, not a live vacancy service.

Test playback, image loading, keyboard access, mobile layout and all links after publication.


## 8. Items that require the project owner's local access or decision

These steps are intentionally not marked complete by a remote repository update:

- Run the commands on the development Mac and report the actual outputs.
- Confirm the locally generated PDF opens and fits on one A4 page.
- Complete the manual Streamlit and Ollama acceptance workflow.
- Capture the genuine screenshots and screen recordings from the running application.
- Confirm the intended deployment experience (public recruiter uploads and inference versus a restricted/recorded demonstration) and acceptable hosting constraints.
- Approve the current resume PDF before any direct-download link is added to the portfolio.
- Review and approve the final public portfolio changes and later YouTube presentation.

A remote source review cannot substitute for these machine- and owner-dependent checks.
