# AI Job-Matching Agent — Technical Walkthrough

## 1. Project Objective

An end-to-end AI/ML application that analyzes candidate resumes and retrieves relevant job opportunities using traditional information retrieval, semantic search, candidate-aware eligibility and local LLM explanations.

## 2. Architecture

Candidate PDF
    |
    v
PyMuPDF Resume Extraction
    |
    v
Structured Candidate Profile
    |
    v
Candidate-Aware Eligibility
    |
    +--------------------+
    |                    |
    v                    v
  TF-IDF            Semantic Search
                         |
                   Text Chunking
                         |
                  Sentence Transformers
                         |
                       FAISS
    |                    |
    +---------+----------+
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

## 3. Data Engineering

Dataset: oksomu/role-radar-dataset

- Original records: 2,500
- Cleaned unique records: 2,471
- Semantic-matching-eligible records: 2,468
- Generated job chunks: 10,095

The pipeline performs location normalization, duplicate removal, description-quality checks, token auditing, chunk generation, embedding generation and FAISS indexing.

Raw data and generated artifacts are excluded from Git.

The dataset was scraped on April 30 and May 1, 2026. It is archived data, not a live vacancy feed.

## 4. Retrieval Engineering

### TF-IDF

Scikit-learn TF-IDF with unigrams and bigrams provides the traditional lexical retrieval baseline.

### Semantic Search

Model: sentence-transformers/all-MiniLM-L6-v2

- Embedding dimensions: 384
- Maximum sequence length: 256 tokens
- Chunk size: 200 tokens
- Overlap: 40 tokens
- Indexed chunks: 10,095

Embeddings are normalized and indexed using FAISS IndexFlatIP. Normalized inner product is equivalent to cosine similarity.

Chunk similarity scores are aggregated by job ID using the maximum score.

### Recommendation Diversification

Near-duplicate jobs are grouped using company, title, location and description similarity. The current threshold is 0.995. Alternate posting IDs are retained.

## 5. Candidate Eligibility

Eligibility filtering happens before retrieval.

It considers:
- User-selected target roles.
- India location constraint.
- Experience requirements.
- Seniority.
- Mandatory advanced education.
- Explicit academic/project equivalency conditions.

Both retrieval methods receive the same eligible job set.

These rules are heuristics and do not guarantee that every job requirement is satisfied.

## 6. Resume Processing

PyMuPDF extracts selectable text from PDF resumes.

Validation:
- Maximum upload size: 10 MB.
- Maximum pages: 20.
- Invalid and password-protected PDFs are rejected.

Rule-based extraction identifies explicit skills, resume sections and education evidence.

Target roles are provided separately by the user and are not inferred from the resume.

OCR for scanned PDFs is not implemented.

## 7. Local LLM Analysis

The application uses Qwen3 4B through local Ollama by default.

Python calculates skill matches, required-skill coverage and skill gaps. The LLM generates supporting explanations.

Pydantic validation and unsupported-claim checks protect the output. Invalid responses trigger deterministic fallback explanations.

Skill coverage measures explicit taxonomy overlap. It is not a hiring probability.

## 8. API and Frontend

FastAPI endpoints:

| Endpoint | Purpose |
|---|---|
| GET /health | Application readiness |
| POST /upload-resume | Resume extraction |
| POST /match | Job retrieval and ranking |
| POST /analyze-job | Selected-job analysis |

Streamlit provides resume upload, target-role selection, candidate evidence, eligibility counters, ranking tabs and selected-job analysis.

The matching engine is initialized during the API application lifecycle and reused across requests.

## 9. Hardware Support

The shared device selector prefers:

1. CUDA
2. Apple MPS
3. CPU

Both embedding generation and runtime matching use this selection policy.

FAISS indexing uses CPU independently of the PyTorch inference device.

Local integration was tested using Apple MPS. CUDA and CPU have not been independently benchmarked on physical hardware.

## 10. Evaluation Results

Manual evaluation:
- 31 reviewed postings.
- 27 recommendation groups.
- 7 relevant groups.

| Metric | TF-IDF | Semantic | Diversified Semantic |
|---|---:|---:|---:|
| Precision@5 | 0.6000 | 0.6000 | 0.6000 |
| Precision@10 | 0.4000 | 0.4000 | 0.4000 |
| Recall@10 within judged groups | 0.5714 | 0.5714 | 0.5714 |
| MRR | 1.0000 | 1.0000 | 1.0000 |
| NDCG@10 | 0.7316 | 0.7657 | 0.7657 |

Semantic retrieval achieved higher NDCG@10 in this evaluation pool, but precision and recall did not improve.

These are preliminary results from one synthetic candidate profile and a limited manually judged pool. Recall is within judged groups, not the complete dataset.

## 11. Automated Testing

Latest full test result:

92 passed, 5 warnings.

Coverage includes:
- Resume parsing and profiling.
- Education extraction.
- Eligibility filtering.
- Retrieval constraints.
- Hardware selection.
- LLM validation and fallback.
- API behavior.

The five warnings are non-blocking SWIG/PyMuPDF deprecation warnings.

## 12. Important Engineering Decisions

| Decision | Reason |
|---|---|
| TF-IDF baseline | Conventional retrieval comparison |
| Token-aware chunking | Avoid silent model-input truncation |
| Normalized embeddings | Cosine-equivalent FAISS search |
| Job-level aggregation | Avoid repeated recommendations |
| Eligibility before retrieval | Exclude known unsuitable jobs |
| Recommendation diversification | Reduce duplicate results |
| Python-owned skill calculations | Deterministic skill coverage |
| Local Ollama | Avoid paid hosted LLM inference |
| FastAPI and Streamlit separation | Separate UI and model lifecycle |
| Automated tests | Protect implementation quality |

## 13. Current Limitations

Not implemented:
- Live job-feed ingestion.
- OCR for scanned resumes.
- User accounts and persistent candidate history.
- Authentication and rate limiting.
- Production deployment and load testing.

Other limitations:
- Eligibility and resume extraction are rule-based.
- Missing explicit skills do not prove missing capability.
- Evaluation is too limited to establish general performance.
- Archived job postings are not verified open vacancies.
- Redistribution rights for third-party job descriptions have not been independently established.


## 14. Recruiter Demonstration

The intended recruiter experience is a recorded walkthrough of the actual locally running Streamlit application, supported by genuine screenshots and the public GitHub source. A continuously hosted public inference service is not required for this portfolio milestone.

A fictional candidate resume is generated reproducibly with:

```bash
python scripts/create_demo_resume.py
```

The PDF is generated from `data/sample/demo_candidate_resume.txt` and contains no real candidate data. The recording should show PDF upload, profile extraction, target-role selection, TF-IDF and semantic recommendation views, and selected-job AI analysis. The exact workflow and acceptance checks are documented in `docs/demo-recording-guide.md`.

The demo must preserve important limitations: the job corpus is archived, eligibility rules are heuristic, and skill coverage is not a hiring probability. Only genuine local application output may be shown.
