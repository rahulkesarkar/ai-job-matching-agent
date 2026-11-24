# Recruiter Demo Recording Guide

## Purpose and agreed demonstration scope

The recruiter experience is a polished portfolio and a public GitHub repository containing source code, genuine screenshots and a video of the **real application running locally**.

The AI application is intentionally not an always-on public service. Recruiters are not expected to upload their own resumes or use the backend themselves. No continuously running server or public LLM inference endpoint is required.

The recording must show a real successful run of the actual Streamlit application. Do not use mock screens, fabricated results, static mock-ups or generated images as evidence of application behavior.

## 1. Safe demonstration data

Use only the synthetic candidate resume in `data/sample/demo_candidate_resume.pdf`.

Generate it from the repository root:

```bash
./.venv314/bin/python scripts/create_demo_resume.py
```

Open the PDF and verify that it is a complete one-page A4 document. The candidate is fictional; the email uses the reserved `example.invalid` domain. Do not use a personal resume or another real person's resume in public screenshots or video.

## 2. Verify the local environment

```bash
./.venv314/bin/python -m pytest -v
./.venv314/bin/python scripts/smoke_test_api.py
```

Confirm that the locally prepared data, embeddings and FAISS index are present. They are intentionally excluded from Git and must be provisioned using the README instructions if missing.

Check Ollama and the configured model:

```bash
ollama list
```

The default model is `qwen3:4b`. Start Ollama locally before the primary AI-analysis recording.

## 3. Start the application locally

Terminal 1 — FastAPI:

```bash
set -a
source .env
set +a
./.venv314/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — Streamlit:

```bash
set -a
source .env
set +a
./.venv314/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Verify `http://127.0.0.1:8000/health` and open `http://127.0.0.1:8501`.

Keep the application bound to `127.0.0.1`. Do not expose the development server to the public internet for this demonstration.

## 4. End-to-end acceptance checklist

1. Upload `data/sample/demo_candidate_resume.pdf`.
2. Enter `Junior Data Analyst` as the target role.
3. Submit the form and wait for processing to finish.
4. Inspect the extracted candidate profile and recognized skills.
5. Confirm eligible and excluded job counts are visible and plausible.
6. Review TF-IDF, semantic and diversified recommendation tabs.
7. Inspect actual job titles and metadata. The corpus is archived; do not imply jobs are currently open.
8. Select a returned job and choose **Analyze this job with local AI**.
9. Wait for Qwen3 analysis and inspect skill matches, gaps, evidence and recommendation.
10. Confirm fallback behavior when Ollama is unavailable or returns invalid output.
11. Restart the services and repeat the core workflow to confirm reproducibility.

Record actual outcomes, errors, latency and resource usage. Do not mark a check complete until it has been performed on the local machine.

## 5. Screenshot plan

Seven genuine screenshots were captured from the real local application using the fictional demonstration candidate. The final JPG files are stored under `docs/demo/screenshots/`:

- `01-resume-upload.jpg` — synthetic resume upload.
- `02-candidate-profile.jpg` — extracted candidate profile.
- `03-tfidf-recommendations.jpg` — lexical retrieval results.
- `04-semantic-recommendations.jpg` — semantic retrieval results.
- `05-diversified-recommendations.jpg` — diversified semantic results.
- `06-ai-job-analysis.jpg` — selected-job analysis.
- `07-supporting-evidence.jpg` — candidate and job evidence.

Before public release, review the final images for private information, credentials, unrelated desktop content and readability. Do not fabricate or alter application results.

## 6. Video recording plan

Create a polished 2–3 minute, 16:9 screen recording of the actual local application:

- 0:00–0:15 — Problem statement and what the prototype demonstrates.
- 0:15–0:40 — Upload the synthetic PDF and select the target role.
- 0:40–1:05 — Show the extracted profile and eligibility filtering.
- 1:05–1:40 — Compare TF-IDF, semantic and diversified recommendations.
- 1:40–2:20 — Analyze one selected job and explain matched skills, gaps and evidence.
- 2:20–2:40 — Close with the architecture, limitations and GitHub repository.

Keep text readable and audio clear. Show genuine loading and processing states. Do not conceal failures or imply a result was live when it was not. If the LLM falls back to deterministic output, explain that honestly.

Save a compressed MP4 under `docs/demo/` when practical. Keep the file reasonably sized for GitHub (preferably below 50 MB); if the recording is larger, use a GitHub-hosted video asset or another stable link from the README rather than committing an oversized binary.

## 7. GitHub evidence package

The repository should make it easy for a recruiter to verify the project:

- `docs/demo/screenshots/` — seven genuine JPG application screenshots.
- `docs/demo/ai-job-matching-agent-demo.mp4` — compressed real screen recording, or a durable video link referenced from the README.
- `README.md` — concise demo section linking to the gallery and video.
- Existing source, tests, architecture, setup, evaluation and limitations documentation.

Add short captions explaining what each screenshot demonstrates. Verify all assets and links from a signed-out browser session.

## 8. Portfolio integration

After the evidence is real and committed:

- Link the project case study to the GitHub repository.
- Add selected screenshots with useful captions.
- Embed or link the actual screen recording.
- Clearly state that the app runs locally for the recorded demonstration and is not a public resume-upload service.
- Retain archived-job-corpus and evaluation caveats.
- Test desktop/mobile layout, image loading, video playback, keyboard access and every link.

## 9. Later presentation

The YouTube presentation is a separate later deliverable. Reuse the verified screen recording and technical narrative only after local acceptance and the repository evidence package are complete.

## 10. Owner-dependent work

These steps require the owner's Mac or approved assets and cannot be completed by editing repository documentation remotely:

- Run the local commands and provide actual test output.
- Generate and inspect the synthetic one-page PDF.
- Complete the manual Streamlit and Ollama workflow.
- Capture genuine screenshots and a real screen recording.
- Review and approve the public evidence.
- Approve the correct current resume PDF before adding a homepage download.

A repository update or CI pass is not a substitute for these local checks.