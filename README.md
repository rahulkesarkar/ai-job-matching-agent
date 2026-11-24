
# AI Job-Matching Agent

An end-to-end AI/ML application that analyzes candidate resumes and retrieves relevant job opportunities using traditional information retrieval, semantic search, candidate-aware eligibility filtering, and locally hosted LLM explanations.

**Project status:** Working local prototype. Evaluated using a small manually judged dataset. Not a live job board or production hiring system.

## Features

- PDF resume extraction using PyMuPDF.
- Structured candidate-profile extraction.
- Explicit skill and education evidence.
- User-selected target roles.
- Candidate-aware job eligibility filtering.
- TF-IDF lexical retrieval baseline.
- Sentence Transformer semantic retrieval.
- FAISS vector search.
- Job-level similarity aggregation.
- Duplicate and near-duplicate recommendation diversification.
- Local Ollama-powered job analysis.
- Deterministic skill coverage and skill-gap calculations.
- FastAPI backend and Streamlit frontend.

## Architecture

```text
Candidate PDF
     |
     v
Resume Extraction
     |
     v
Candidate Profile
     |
     v
Candidate Eligibility
     |
     +----------------------+
     |                      |
     v                      v
 TF-IDF              Semantic Retrieval
                            |
                      Text Chunking
                            |
                   Sentence Transformers
                            |
                          FAISS
     |                      |
     +----------+-----------+
                |
                v
       Job-Level Aggregation
                |
                v
    Recommendation Diversification
                |
                v
             FastAPI
                |
                v
           Streamlit
                |
                v
        Selected Job Analysis
                |
                v
       Python Validation + Ollama
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy, PyArrow, DuckDB |
| Lexical retrieval | Scikit-learn |
| Embeddings | Sentence Transformers, PyTorch |
| Vector search | FAISS |
| Resume parsing | PyMuPDF |
| Local LLM | Ollama, Qwen3 |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Validation | Pytest |

## Retrieval Engineering

### TF-IDF baseline

Scikit-learn TF-IDF with unigrams and bigrams provides a traditional lexical retrieval baseline.

### Semantic retrieval

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

- Embedding dimensions: 384
- Maximum sequence length: 256 tokens
- Chunk size: 200 tokens
- Chunk overlap: 40 tokens
- Generated job chunks: 10,095

Embeddings are normalized and indexed using FAISS `IndexFlatIP`. Normalized inner-product similarity is equivalent to cosine similarity.

Chunk-level scores are aggregated by job ID using the maximum similarity score.

### Recommendation diversification

Near-duplicate postings are grouped using normalized company, title, location and description similarity. The current description similarity threshold is 0.995. Alternative posting IDs are retained.

### Candidate eligibility

Eligibility rules run before retrieval. They consider target roles, location, experience, seniority and education evidence. Both retrieval methods receive the same eligible job set.

These rules are heuristics and do not guarantee that every job-specific requirement is satisfied.

## LLM Analysis

The application uses local Ollama inference, with `qwen3:4b` as the default model.

Python calculates required-skill matches, coverage and gaps. The LLM provides a supporting explanation.

Structured validation and unsupported-claim checks protect the output. Invalid responses trigger a deterministic fallback.

Skill coverage represents explicit taxonomy overlap, not a probability of getting an interview or job offer.

## Dataset

Source: [Role Radar Dataset](https://huggingface.co/datasets/oksomu/role-radar-dataset)

Processing results:

| Stage | Records |
|---|---:|
| Original dataset | 2,500 |
| Cleaned unique jobs | 2,471 |
| Semantic-matching-eligible jobs | 2,468 |
| Generated job chunks | 10,095 |

The dataset was scraped on April 30 and May 1, 2026. It is an archived research dataset, not a live vacancy feed.

Raw job data and generated artifacts are excluded from Git. The repository tracks the location alias reference file separately.

The dataset card advertises Apache-2.0, but redistribution rights for individual third-party job descriptions have not been independently established. Raw data is therefore not committed.

## Repository Structure

```text
app/
  main.py
  streamlit_app.py
  services/

data/
  raw/
  processed/
  reference/
  sample/

docs/
  engineering-notes.md
  project-walkthrough.md
  relevance-judgment-guidelines.md

notebooks/
  01-28 data and ML experiments

scripts/
  prepare_data.py
  smoke_test_api.py
  smoke_test_llm_analysis.py

tests/
  automated regression tests
```

## Setup

### Prerequisites

- Python 3.14.3
- Git
- Ollama
- Internet access for package, dataset and model downloads

### 1. Clone the repository

```bash
git clone https://github.com/rahulkesarkar/ai-job-matching-agent.git
cd ai-job-matching-agent
```


### 2. Create an environment

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The device-selection utility prefers CUDA, then Apple MPS, then CPU.

### 3. Download the dataset

Create the raw-data directory:

```bash
mkdir -p data/raw
```

Download the source file:

```bash
hf download oksomu/role-radar-dataset \
    scraped_jobs.json \
    --repo-type dataset \
    --local-dir data/raw
```

Rename it to the filename expected by the preprocessing code:

```bash
mv data/raw/scraped_jobs.json \
   data/raw/role_radar_scraped_jobs.json
```

### 4. Prepare the dataset and search index

Validate the preparation inputs:

```bash
python scripts/prepare_data.py
```

Run the complete pipeline:

```bash
python scripts/prepare_data.py --run
```

The pipeline performs:

1. Location normalization.
2. Job dataset cleaning.
3. Token-length auditing.
4. Chunk-volume estimation.
5. Job chunk generation.
6. Embedding generation.
7. FAISS index construction.

**Note:** The `--run` command regenerates processed datasets, embeddings and the FAISS index. The model is downloaded from Hugging Face when it is not already cached.

### 5. Configure Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download the model:

```bash
ollama pull qwen3:4b
```

Check that the model is installed:

```bash
ollama list
```

The default configuration uses local inference without a paid hosted LLM API.

### 6. Configure environment variables

Create the local configuration file:

```bash
cp .env.example .env
```

Load it in each terminal session:

```bash
set -a
source .env
set +a
```

The configuration supports:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OLLAMA_NUM_CTX`
- `OLLAMA_NUM_PREDICT`
- `OLLAMA_TIMEOUT_SECONDS`
- `JOB_MATCH_API_URL`

The application reads environment variables directly. The `.env` file is not automatically loaded by Python.

Do not commit `.env`.

### 7. Start FastAPI

In Terminal 1:

```bash
source .venv/bin/activate

set -a
source .env
set +a

python -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload
```

API health endpoint:

http://127.0.0.1:8000/health

Interactive API documentation:

http://127.0.0.1:8000/docs

### 8. Start Streamlit

In Terminal 2:

```bash
source .venv/bin/activate

set -a
source .env
set +a

streamlit run app/streamlit_app.py \
    --server.address 127.0.0.1 \
    --server.port 8501
```

Open:

http://127.0.0.1:8501

Upload a selectable-text PDF resume, select target roles, review eligible job counts and compare the recommendation methods.

## Evaluation Results

The retrieval comparison used 31 manually reviewed postings, grouped into 27 recommendation groups, with 7 relevant groups.

| Metric | TF-IDF | Semantic | Diversified Semantic |
|---|---:|---:|---:|
| Precision@5 | 0.6000 | 0.6000 | 0.6000 |
| Precision@10 | 0.4000 | 0.4000 | 0.4000 |
| Recall@10 within judged groups | 0.5714 | 0.5714 | 0.5714 |
| MRR | 1.0000 | 1.0000 | 1.0000 |
| NDCG@10 | 0.7316 | 0.7657 | 0.7657 |

Semantic retrieval achieved higher NDCG@10 in this evaluation pool, but precision and recall did not improve.

These results are preliminary and based on one synthetic candidate profile and a limited manually judged dataset. Recall is within judged groups, not across the complete corpus.

## Automated Testing

Run the full test suite locally:

```bash
python -m pytest -v
```

GitHub Actions also runs the Python 3.14 regression suite on pushes and pull requests targeting `main`. The workflow installs the pinned development dependencies, compiles the application/scripts/tests, and runs the full pytest suite. A configured workflow is not evidence of a passing run; check the latest Actions result for the commit being reviewed.

Previously recorded local result (historical):

```text
92 passed, 5 warnings
```

A fresh GitHub Actions run on November 24, 2026, for commit `9ecdbfe6bf3885bbc4a379bee36ca77264c3e1a0` completed successfully on Python 3.14. The workflow installed dependencies, compiled the application/scripts/tests, and ran the automated regression suite. This verifies the GitHub Ubuntu runner; local Mac end-to-end verification is still required. The earlier local run reported non-blocking SWIG/PyMuPDF deprecation warnings.

## Local Verification

Fresh local verification on November 24, 2026, was performed on Apple Silicon with Python 3.14.3:

- **92 automated tests passed** (5 non-blocking SWIG/PyMuPDF deprecation warnings).
- FastAPI integration smoke test passed, including actual PDF parsing, candidate profiling, eligibility filtering and all three retrieval methods.
- The matching engine loaded 2,468 jobs, 10,095 indexed chunks and 384-dimensional embeddings on Apple MPS.
- Local Ollama analysis passed using `qwen3:4b`.
- The Streamlit application completed a manual local resume-upload and recommendation workflow.
- The live UI deterministic fallback was verified during an Ollama outage, and normal Qwen3 analysis was verified after Ollama restarted.

These checks establish that the local prototype ran end to end on the development Mac. They do not establish live-job freshness, broad retrieval quality, production readiness or public availability. The corpus remains archived, and evaluation remains preliminary.

## Security and Limitations

- Resume PDFs are processed in memory rather than persisted by the resume parser.
- The default LLM workflow uses local Ollama inference.
- Resume evidence is selected and redacted before LLM analysis.
- This is a local demonstration prototype, not a public resume-upload service; keep development servers bound to localhost.
- Public-service authentication, rate limiting and load testing are not implemented because always-on public access is outside the agreed scope.
- OCR for scanned PDFs is not implemented.
- Resume extraction and eligibility rules are heuristic.
- Missing explicit skills do not prove that a candidate lacks the capability.
- The dataset does not represent verified live vacancies.
- Third-party job-description redistribution rights have not been independently established.

## Documentation

- `docs/engineering-notes.md` — implementation history and engineering decisions.
- `docs/project-walkthrough.md` — technical architecture and recruiter walkthrough.
- `docs/relevance-judgment-guidelines.md` — manual evaluation methodology.

## License

A project-level license has not yet been selected. Review the upstream dataset licensing and third-party content redistribution rights before publishing data artifacts.


## Recruiter Demonstration

The agreed recruiter experience is a polished portfolio case study leading to the public GitHub repository, where recruiters can inspect the source, genuine application screenshots and a recorded video of the **real application running locally**. The application is not intended to be an always-on public service; recruiters do not need to upload their own resumes or access a live backend.

Generate the synthetic candidate resume:

```bash
./.venv314/bin/python scripts/create_demo_resume.py
```

The candidate is fictional and the email uses the reserved `example.invalid` domain. Use only this synthetic resume in public screenshots and recordings.

Follow [the demo recording guide](docs/demo-recording-guide.md) for local setup, acceptance checks, screenshot capture, recording and the GitHub evidence package. After local verification, add genuine screenshots under `docs/demo/screenshots/` and a compressed MP4 or stable video link under `docs/demo/`; link both from this README.

**Demonstration scope:** local PDF upload, profile extraction, target-role selection, TF-IDF and semantic recommendations, diversified results, and selected-job analysis using local Ollama. The job corpus is archived and does not represent verified current vacancies. Keep the local app bound to localhost; public deployment and 24/7 hosting are explicitly out of scope.
