
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

Run the full test suite:

```bash
python -m pytest -v
```

Latest verified result:

```text
92 passed, 5 warnings
```

The warnings are non-blocking SWIG/PyMuPDF deprecation warnings.

## Security and Limitations

- Resume PDFs are processed in memory rather than persisted by the resume parser.
- The default LLM workflow uses local Ollama inference.
- Resume evidence is selected and redacted before LLM analysis.
- Authentication and rate limiting are not implemented.
- OCR for scanned PDFs is not implemented.
- Resume extraction and eligibility rules are heuristic.
- Missing explicit skills do not prove that a candidate lacks the capability.
- The dataset does not represent verified live vacancies.
- Production deployment and load testing have not been completed.
- Third-party job-description redistribution rights have not been independently established.

## Documentation

- `docs/engineering-notes.md` — implementation history and engineering decisions.
- `docs/project-walkthrough.md` — technical architecture and recruiter walkthrough.
- `docs/relevance-judgment-guidelines.md` — manual evaluation methodology.

## License

A project-level license has not yet been selected. Review the upstream dataset licensing and third-party content redistribution rights before publishing data artifacts.
