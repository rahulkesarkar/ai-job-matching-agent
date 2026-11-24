# Engineering Notes — AI Job-Matching Agent

## 1. Development Environment

Operating System: macOS 26.3
Processor Architecture: Apple Silicon ARM64
Python Version: 3.14.3
Virtual Environment: .venv314

### Why use a virtual environment?

A virtual environment isolates the dependencies of
one Python project from other projects and the system.

This prevents package-version conflicts and makes
the development environment easier to reproduce.

Activate the environment:

source .venv314/bin/activate

## 2. Project Architecture

app/api       - Backend API endpoints
app/core      - Configuration and application settings
app/models    - Data models and schemas
app/services  - Business logic and ML components
app/utils     - Reusable helper functions

data/raw      - Original datasets
data/processed - Cleaned datasets
data/sample   - Small test datasets

notebooks     - Data analysis and ML experiments
tests         - Automated tests
docs          - Technical documentation

## 3. Planned Data Engineering Tools

Pandas:
Used for DataFrames, data cleaning, transformation,
missing-value analysis and exploratory data analysis.

PyArrow:
Used for columnar data formats, especially Parquet,
which is useful for storing large structured datasets.

DuckDB:
A lightweight analytical database that allows SQL
queries directly against data files such as Parquet.

## 4. Planned Machine Learning Approach

TF-IDF:
Our traditional text-matching baseline.

Sentence Transformers:
Used to convert resumes and job descriptions into
numerical vector representations called embeddings.

FAISS:
Used to search for similar vectors efficiently.

Hybrid Ranking:
Combines semantic similarity with other signals
such as skill overlap and experience requirements.

Evaluation:
Precision@K, Recall@K, MRR and NDCG will be used
to evaluate the quality of job retrieval.

## 5. Engineering Principles

- Keep raw and processed data separate.
- Never upload API keys or private resumes.
- Record dataset sources and licences.
- Validate data before using it.
- Test each major component.
- Document technical decisions and trade-offs.
- Never claim unfinished features are implemented.

## 6. Current Progress

Completed:
- Python environment setup
- Virtual environment verification
- Initial project directory structure
- Git repository initialization
- Initial README

Pending verification:
- First Git commit

Not yet implemented:
- Dataset ingestion
- Data preprocessing
- Resume parsing
- TF-IDF matching
- Embedding generation
- FAISS retrieval
- Hybrid ranking
- LLM integration
- FastAPI backend
- Streamlit interface
- Docker and AWS deployment

## 7. Data Engineering Dependencies — Installed

Python version: 3.14.3

Verified package versions:

- Pandas: 3.1.1
- PyArrow: 26.0.1
- DuckDB: 2.0.1
- NumPy: 2.5.5
- python-dateutil: 4.0.0
- tzdata: 2026.6

### Why these libraries?

Pandas:
Provides DataFrame operations for cleaning and
analyzing structured job data.

PyArrow:
Supports columnar data formats such as Parquet,
which are useful for efficient storage and retrieval.

DuckDB:
Provides analytical SQL queries over datasets,
including Parquet files.

NumPy:
Provides numerical arrays and computation used
by many Python data science libraries.

### Dependency management

The command:

python -m pip freeze > requirements.txt

records installed package versions.

The command:

python -m pip install -r requirements.txt

checks that the declared dependencies can be resolved.

A clean-environment installation test will be
performed separately when needed.

### Verification

All four primary libraries were successfully imported.
The installed package versions were printed and checked.
pip check will also be used to detect dependency conflicts.


## 8. Initial Dataset Quality and Coverage Findings

Dataset: Role Radar job postings
Source: https://huggingface.co/datasets/oksomu/role-radar-dataset
Original records: 2,500
Fields: 16

### Data quality findings

- 29 duplicate job IDs were found.
- All 29 duplicate ID groups contain identical records.
- 3 columns are entirely empty:
  posted_at, domain_hint, experience_years_hint.
- 3 job descriptions contain fewer than 100 characters.
- 136 distinct descriptions are repeated.
- 340 records belong to repeated-description groups.
- Only 5 repeated-description groups occur across
  multiple companies.

Repeated descriptions are not automatically duplicates.
Different job records may legitimately share templates.

### Geographic coverage

- 1,915 records explicitly contain the word India.
- 584 non-empty records do not explicitly mention India.
- 1 location is missing.

Some Indian metropolitan locations do not include
the word India.

Known Indian location aliases in the non-India group:

- Mumbai Metropolitan Region: 184
- Greater Kolkata Area: 55
- Pune/Pimpri-Chinchwad Area: 28
- Greater Bengaluru Area: 13
- Greater Ahmedabad Area: 13
- Greater Chennai Area: 9
- Greater Hyderabad Area: 6

These account for 308 additional records.

At least 2,223 records can therefore be associated
with India using the current location evidence.

This is a preliminary count, not a final geographic
classification. Remaining locations require inspection.

Engineering decision:
Build a location normalization pipeline instead of
using simple string matching for country detection.

### Job-role coverage

Technology-related role families:

- Fullstack: 312
- Data: 244
- Backend: 235
- Frontend: 164
- DevOps: 145
- Mobile: 74

Combined: 1,174 records (46.96%).

### Seniority coverage

- Entry level: 176
- Internship: 100
- Associate: 458
- Mid-Senior: 562
- Director: 224
- Executive: 11
- Not Applicable: 969

Entry-level and internship records total 276.

Associate roles will not automatically be classified
as fresher opportunities. Job descriptions must be
checked for their actual experience requirements.

### Data freshness

The dataset was scraped on:
- 2026-04-30: 566 records
- 2026-05-01: 1,934 records

This is an archived dataset, not a live job feed.

Engineering decision:
Use it for offline retrieval experiments and model
evaluation, not for claiming current job availability.

### Dataset handling

The dataset card advertises an Apache 2.0 licence.
However, individual job descriptions may contain
third-party employer content.

We will not commit the complete raw dataset to GitHub
until its redistribution rights have been verified.

### Current limitations

- Geographic classification is not yet finalized.
- Job experience requirements need extraction.
- Job descriptions need quality validation.
- Some metadata columns contain no usable information.
- The dataset is not current enough for live vacancy claims.


## 9. Location Normalization — First Implementation

Created:
- notebooks/04_location_audit.py
- notebooks/05_normalize_locations.py
- data/processed/location_aliases.csv
- data/processed/jobs_with_normalized_locations.parquet

### Initial results

Total job records: 2,500

Alias-mapped records: 316

Preliminarily identified Indian records: 2,231

Unresolved locations: 268

Missing locations: 1

### Implementation approach

Created a separate CSV containing manually reviewed
location aliases.

Used Pandas merge() with:
- how="left" to preserve every original job record.
- validate="many_to_one" to prevent unexpected
  duplicate rows caused by the mapping table.

Explicitly Indian locations were identified using
their country suffix.

Unrecognized locations were retained as unresolved
rather than being assigned an assumed country.

### Why use this approach?

Location names can represent cities, districts,
metropolitan areas and regions.

Simple substring matching is insufficient for
reliable geographic classification.

Keeping aliases in a separate CSV makes the mapping
maintainable without changing application code.

### Parquet storage

The normalized dataset was saved in Parquet format
using PyArrow.

Parquet is a columnar storage format that supports
efficient analytical queries and is suitable for
larger datasets.

### Current limitations

- Some Indian city names are not yet canonicalized.
- Locations without recognized country information
  remain unresolved.
- Geographic classification is preliminary.
- No job records have been deleted.
- Original job descriptions and IDs are retained.

Next:
Validate Parquet integrity and inspect the processed
dataset using DuckDB.

## 10. Parquet Validation Using DuckDB

Created:
- notebooks/06_verify_parquet.py

Processed file:
- data/processed/jobs_with_normalized_locations.parquet

### Validation results

Original records: 2,500
Parquet records: 2,500
Unique job IDs: 2,471
Duplicate job IDs: 29
Parquet file size: 3.8 MB

Country classification:
- India: 2,231
- Unclassified: 269

### DuckDB implementation

Used DuckDB with an in-memory connection.

The read_parquet() function allowed SQL queries
directly against the Parquet file.

SQL operations used:
- COUNT(*) to count all records.
- COUNT(DISTINCT id) to count unique job IDs.
- GROUP BY to examine normalization status
  and country distribution.

### Why DuckDB?

It allows analytical queries against Parquet
without first loading the entire dataset into
a Pandas DataFrame.

This becomes useful when working with larger
job datasets.

### Validation conclusion

All 2,500 original job records survived
the location-normalization process.

The raw dataset remains unchanged.

The 29 duplicate job-ID records were already
verified as identical in the previous investigation.

Next:
Create a separate cleaned dataset with duplicate
job records removed and validate the result.


## 11. Data Cleaning Pipeline — Verified Results

Created:
- notebooks/07_clean_dataset.py
- data/processed/jobs_cleaned.parquet

### Cleaning operations

1. Checked for missing job IDs.
2. Checked duplicate IDs for conflicting records.
3. Removed verified duplicate job records.
4. Removed completely empty columns.
5. Measured job description lengths.
6. Classified description quality.
7. Created semantic matching eligibility flags.
8. Saved the cleaned dataset in Parquet format.
9. Reloaded the output and validated the results.

### Verified results

Original records: 2,500
Cleaned records: 2,471
Duplicate records removed: 29
Unique job IDs: 2,471
Remaining duplicate IDs: 0

Description quality:
- Usable: 2,468
- Short review: 2
- Placeholder: 1

Semantic matching eligibility:
- Eligible: 2,468
- Not eligible: 3

Missing job functions: 7

Description length:
- Mean: 3,285.95 characters
- Median: 2,812 characters
- Maximum: 18,886 characters

### Engineering decisions

Duplicate records were removed using job IDs because
the duplicate groups had already been verified as
identical.

Completely empty columns were removed because they
provided no usable information.

Short and placeholder descriptions were retained
but flagged rather than silently deleted.

Missing job functions were preserved rather than
filled with invented classifications.

### Important distinction

A retained job record is not automatically eligible
for semantic matching.

The eligibility flag allows downstream components
to filter records according to description quality.

The 100-character threshold is an initial heuristic
that may be revised after further evaluation.

### Validation

All 2,471 retained job IDs are unique.
The saved Parquet file was reloaded successfully.
The record count and description quality statistics
were independently checked.


## 12. First Machine Learning Experiment: TF-IDF

Created:
- notebooks/08_tfidf_matching.py
- notebooks/09_inspect_tfidf_results.py
- data/sample/sample_candidate_profile.txt
- data/processed/tfidf_ranked_jobs.csv

### Candidate input

Used a synthetic candidate profile containing:
- Target job roles
- Technical skills
- Project experience
- Career preferences

Location and employment preferences were excluded
from TF-IDF text matching so they would not distort
technical relevance.

### Model implementation

Algorithm: TF-IDF
Similarity metric: Cosine similarity
Library: Scikit-learn

Vectorizer configuration:
- lowercase=True
- stop_words="english"
- ngram_range=(1, 2)
- min_df=1
- max_df=0.95
- sublinear_tf=True

The vocabulary and inverse document frequencies
were learned from eligible job documents.
The candidate profile was transformed using
the same fitted vectorizer.

### First experiment results

Jobs compared: 2,468

Similarity score statistics:
- Mean: 0.0122
- Median: 0.0093
- 75th percentile: 0.0140
- Maximum: 0.1540
- Minimum: 0.0000

### Top retrieved jobs

1. Data Scientist - InfoCepts
   Score: 0.1540
   Metadata seniority: Entry level

2. Junior AI & Data Engineer - OmniReach
   Score: 0.1499
   Metadata seniority: Mid-Senior level
   Description explicitly mentions freshers
   and early-career engineers.

3. Data Scientist - ION
   Score: 0.1297
   Title mentions a PhD program.
   Qualification suitability needs investigation.

Other retrieved results included senior and
director-level data science and AI/ML positions.

### Observed limitations

1. TF-IDF measures lexical similarity, not
   actual candidate suitability.

2. Senior and director-level jobs can rank highly
   because they share technical vocabulary.

3. Seniority metadata can conflict with the
   actual wording of job descriptions.

4. Similar job descriptions can produce nearly
   identical recommendations.

5. Similarity scores are not probabilities and
   cannot be interpreted as hiring chances.

6. The current candidate profile is synthetic
   and is only a prototype evaluation input.

### Important next step

Create an independent relevance-judgment dataset.

We need human-reviewed relevance labels before
calculating Precision@K, Recall@K, MRR or NDCG.

We must not evaluate the model using its own
similarity scores as ground truth.


## 13. TF-IDF Pilot Evaluation — Actual Results

Created:
- notebooks/11_apply_relevance_judgments.py
- notebooks/12_evaluate_tfidf.py
- docs/relevance-judgment-guidelines.md
- data/processed/tfidf_top10_for_review.csv
- data/processed/tfidf_pilot_metrics.json

### Human relevance judgments

Total reviewed jobs: 10

- Grade 2 (highly relevant): 3
- Grade 1 (partially relevant): 1
- Grade 0 (not relevant): 6

Binary relevance threshold: grade >= 1

### Measured metrics

Precision@5: 0.6000
Precision@10: 0.4000
Recall@5 within reviewed pool: 0.7500
Recall@10 within reviewed pool: 1.0000
MRR: 1.0000
NDCG@10: 0.9393

### Interpretation

Three of the first five recommendations were relevant.

Four of the first ten recommendations were relevant.

The first recommendation was relevant, resulting
in an MRR of 1.0 for this pilot.

NDCG@10 was 0.9393, indicating that the judged
high-relevance jobs appeared relatively high
within this particular ranked list.

### Evaluation limitations

This is a pilot evaluation based on the first
ten TF-IDF recommendations.

The evaluation pool was selected from the
TF-IDF model's own output.

Therefore:
- Recall@10 of 1.0 is not a full-dataset recall result.
- The MRR result is specific to these ten judgments.
- NDCG is not a general performance guarantee.
- Ten manually judged jobs are insufficient to
  establish statistically reliable model performance.

A broader independent relevance pool is needed.

### Next evaluation improvement

Construct an evaluation pool containing:
- TF-IDF retrieved jobs.
- Independently sampled job records.
- Results from future semantic retrieval models.

Have job relevance assessed independently of
model similarity scores.

Use the same candidate profile and relevance
guidelines for fair model comparisons.


## 14. Sentence Transformers — First Embedding Experiment

Created:
- notebooks/13_test_embedding_model.py

Model:
sentence-transformers/all-MiniLM-L6-v2

Installed libraries:
- Sentence Transformers: 6.3.0
- PyTorch: 2.15.0
- Transformers: 5.24.0

Hardware:
- Apple Silicon ARM64
- PyTorch MPS availability: True
- Model configured to use MPS

### Model characteristics

- Embedding dimension: 384
- Architecture: MiniLM transformer
- Maximum sequence length: 256 word pieces

### Experiment

Three test sentences were converted into
384-dimensional normalized embeddings.

Related sentences:
"Cleaning datasets using Python and SQL."
"Preprocessing and transforming data using Python and SQL."

Unrelated sentence:
"Developing mobile applications using Swift and iOS."

Measured cosine similarities:
- Related pair: 0.5790
- Unrelated pair: 0.0837

### Interpretation

The related sentences had substantially higher
similarity than the unrelated example.

This confirms that the model can produce useful
semantic representations for our initial test.

This is a smoke test, not a formal benchmark
or proof of job-matching accuracy.

### Important limitation: Input truncation

The model supports a maximum input length of
256 word pieces.

Our job descriptions average approximately
3,286 characters, so many descriptions exceed
the model's input limit.

Passing entire job descriptions directly into
the model could silently discard important
requirements near the end.

Next engineering task:
Implement text chunking with token-aware limits
and controlled overlap before generating
the complete job embedding dataset.


## 15. Long Document Chunking — Validation

Created:
- app/services/text_chunker.py
- notebooks/14_audit_token_lengths.py
- notebooks/15_test_text_chunker.py

### Token audit findings

Eligible job descriptions: 2,468
Mean token count: 615.37
Median token count: 529.5
90th percentile: 1,138.3
99th percentile: 1,754.64
Maximum: 3,161 tokens

Jobs exceeding 256 tokens: 2,123 (86.02%)

### Chunking configuration

Embedding model:
all-MiniLM-L6-v2

Maximum model input:
256 tokens

Content chunk size:
200 tokens

Overlap:
40 tokens

Stride:
160 tokens

### Validation results

Tested three real job descriptions:
- Median-length description: 4 chunks
- Longest description: 20 chunks
- Shortest eligible description: 1 chunk

Verified:
- Complete original token-position coverage.
- Exactly 40 overlapping tokens between chunks.
- Every chunk stays within the 256-token model limit.
- Special tokens are included in model-input validation.

### Tokenizer warning investigation

A 528-token document triggered a warning because
the tokenizer's generic configured threshold is 512.

This warning occurs during full-document tokenization.
The complete document is not passed to the model.

The actual chunk inputs remain below the model's
256-token limit.

The warning is documented rather than suppressed.

### Engineering decision

Long descriptions must be chunked before embedding.
This prevents silent loss of job requirements that
appear beyond the model's input limit.

Chunk boundaries and overlap are validated
before embedding generation.


## 16. Chunk Volume Estimation — Actual Results

Created:
- notebooks/16_estimate_chunk_volume.py
- data/processed/chunk_volume_estimate.csv

### Configuration

Eligible jobs: 2,468
Chunk content size: 200 tokens
Overlap: 40 tokens
Stride: 160 tokens
Model input limit: 256 tokens
Embedding dimensions: 384

### Actual workload estimate

Total estimated chunks: 10,095
Average chunks per job: 4.09
Maximum chunks for one job: 20

### Embedding storage estimate

Assumption:
Each embedding contains 384 float32 values.

Calculation:
10,095 × 384 × 4 bytes

Estimated vector storage:
Approximately 14.79 MB.

This estimate excludes metadata, Parquet storage
overhead and vector-index overhead.

### Engineering decision

The workload is suitable for local batch inference
on the available Apple Silicon environment.

We will generate embeddings in batches rather
than trying to process the complete collection
as one inference operation.

### Retrieval consideration

A job can have multiple chunk embeddings.

The retrieval layer must aggregate chunk-level
similarity scores by job ID before presenting
ranked job recommendations.

Otherwise, jobs with more chunks could occupy
multiple positions in the recommendation list.

Next:
Generate and validate the complete chunk dataset,
then produce normalized 384-dimensional embeddings.


## 17. Complete Job Chunk Dataset — Verified

Created:
- notebooks/17_generate_job_chunks.py
- data/processed/job_chunks.parquet

### Actual generation results

Eligible jobs processed: 2,468
Total chunks generated: 10,095
Unique job IDs: 2,468
Unique chunk IDs: 10,095
Duplicate chunk IDs: 0

Maximum observed model input length: 206 tokens
Allowed model input length: 256 tokens

Chunk dataset size: 4.07 MB

### Validation results

- Every eligible job is represented.
- Every chunk has a unique identifier.
- Original job IDs are preserved.
- Chunk positions and metadata are preserved.
- Consecutive chunks have the configured overlap.
- No chunk exceeds the model input limit.

### Storage decision

Chunk text and metadata are stored in Parquet.

Embeddings will be stored separately as NumPy
float32 arrays.

This keeps descriptive metadata separate from
numerical vectors and allows efficient FAISS indexing.

Next:
Generate normalized embeddings for all 10,095
chunks using the pretrained Sentence Transformer.


## 18. Embedding Generation — Batch Test

Created:
- notebooks/18_test_embedding_batch.py

### Test configuration

Model: all-MiniLM-L6-v2
Device: Apple MPS
Test chunks: 64
Inference batch size: 16
Embedding dimensions: 384
Data type: float32
Normalization: Enabled

### Validation results

- Model loaded successfully.
- MPS device explicitly selected and verified.
- Expected embedding shape: (64, 384).
- All embedding values were finite.
- All vectors passed unit-norm validation.
- Batch generation completed successfully.

### Engineering decisions

Embeddings are generated in batches to control
inference memory consumption.

Normalized embeddings allow cosine similarity
to be calculated using vector dot products.

The smoke-test embeddings were kept in memory
and were not saved as the final dataset.

### Next step

Generate embeddings for all 10,095 job chunks.

Store numerical vectors separately from their
job and chunk metadata, with a persistent index
that preserves the exact row-to-chunk mapping.


## 19. Full Semantic Embedding Generation — Completed

Created:
- notebooks/19_generate_embeddings.py
- data/processed/job_chunk_embeddings.npy
- data/processed/job_chunk_embedding_metadata.parquet
- data/processed/embedding_manifest.json

### Model configuration

Model: sentence-transformers/all-MiniLM-L6-v2
Embedding dimensions: 384
Inference device: Apple MPS (mps:0)
Batch size: 16
Data type: float32
Vector normalization: Enabled

### Generation results

Total chunks embedded: 10,095
Unique jobs represented: 2,468
Embedding matrix shape: (10095, 384)
Embedding array size: 14.79 MB
Duplicate chunk IDs: 0

Generation time: Approximately 71 seconds.

### Validation results

- All embedding values are finite.
- Every embedding has 384 dimensions.
- All vectors have unit norm.
- All chunk IDs are unique.
- Metadata contains 10,095 records.
- Embedding-to-metadata row mapping verified.
- Saved embeddings successfully reloaded.

### Storage architecture

Numerical vectors:
job_chunk_embeddings.npy

Job and chunk metadata:
job_chunk_embedding_metadata.parquet

Model configuration:
embedding_manifest.json

### Engineering decision

Vectors and metadata are stored separately.

This allows numerical similarity search to operate
on the embedding matrix while application logic
uses the metadata to identify the corresponding
job title, company, location and job ID.

The row order is preserved so that embedding row i
always corresponds to metadata row i.

### Next component

FAISS vector indexing and semantic retrieval.


## 20. FAISS Vector Search — First Index

Created:
- notebooks/20_build_faiss_index.py
- data/processed/faiss_job_chunks.index
- data/processed/faiss_index_manifest.json

### Index configuration

Library: FAISS
Version: 1.16.0
Index type: IndexFlatIP
Distance metric: Inner product
Search type: Exact similarity search

### Indexed data

Total vectors: 10,095
Vector dimensions: 384
Data type: float32
Embedding model: all-MiniLM-L6-v2
Vectors normalized: Yes

### Why IndexFlatIP?

IndexFlatIP performs exact inner-product searches.

Because our embeddings are normalized to unit length,
inner product is equivalent to cosine similarity.

We selected exact search because our current collection
contains 10,095 vectors, which is manageable locally.

Approximate nearest-neighbour indexing can be explored
if the collection grows substantially.

### Hardware architecture

Embedding generation:
Apple MPS through PyTorch.

Vector indexing and retrieval:
FAISS CPU index.

These are separate components.

### Validation

- All embedding inputs validated.
- Vector dimensions verified.
- Vector normalization verified.
- All 10,095 vectors added to the index.
- Saved index successfully reloaded.
- Test similarity search completed successfully.

### Vector-to-metadata mapping

FAISS vector ID corresponds to the row position
of the associated chunk metadata.

The metadata preserves the original job ID and
chunk ID for retrieval and recommendation display.

Next:
Build the candidate-to-job semantic retrieval pipeline.


## 21. Semantic Retrieval — Duplicate Recommendation Audit

Created:
- notebooks/22_audit_semantic_duplicates.py

### Investigation results

Top 10 semantic recommendations: 10

Exact duplicate description groups: 3

Postings belonging to duplicate groups: 6

### Duplicate detection methodology

Used:
- HTML entity normalization.
- Whitespace normalization.
- SHA-256 description fingerprints.
- Grouping by company, title and description fingerprint.

### Findings

Multiple job IDs can represent postings with
identical descriptions.

These are different from duplicate job IDs
identified during initial data cleaning.

### Engineering decision

Introduce recommendation-level diversification.

Exact duplicate postings will be grouped during
recommendation ranking.

Only the highest-ranked representative will
appear in the diversified result list.

The original job records, IDs and descriptions
will remain unchanged.

This is a ranking-stage operation, not destructive
dataset cleaning.

### Limitation

Identical descriptions do not always prove that
two listings represent the same vacancy.

The deduplication rule is a recommendation-quality
heuristic and will be evaluated against the
candidate's location preferences.


## 22. Recommendation Diversification — Actual Results

Created:
- app/services/recommendation_diversifier.py
- notebooks/23_apply_recommendation_diversification.py
- data/processed/semantic_ranked_jobs_diversified.csv

### Actual results

Original ranked postings: 2,468
Recommendation representatives: 2,326
Additional postings grouped: 142

All original job IDs were accounted for.

### Diversification methodology

- Normalize company names.
- Normalize job titles.
- Normalize HTML entities and whitespace in descriptions.
- Generate SHA-256 description fingerprints.
- Group records using company, title and description.
- Keep the highest-scoring representative.
- Preserve alternate job IDs for traceability.

### Observations

The duplicated Turing recommendation was consolidated.

Two SLB recommendations remain in the top ten.
Their descriptions are nearly identical but differ
after exact text normalization.

Exact matching therefore does not identify
all near-duplicate opportunities.

### Validation

- No duplicate representative job IDs.
- All original job IDs accounted for.
- Duplicate group counts reconcile with input records.
- Original semantic ranking remains unchanged.

### Limitations

Identical descriptions do not necessarily indicate
identical vacancies.

Location was not included in the exact grouping key.
This needs further investigation because postings
in different cities can be distinct opportunities.

Near-duplicate matching may be explored later,
but fuzzy grouping must be validated before
removing recommendations.


## 23. Location-Aware Recommendation Diversification

Updated:
- app/services/recommendation_diversifier.py
- notebooks/23_apply_recommendation_diversification.py
- data/processed/semantic_ranked_jobs_diversified.csv

### Updated grouping criteria

1. Normalized company name.
2. Normalized job title.
3. Exact normalized description fingerprint.
4. Normalized city and country.

When normalized geography is unavailable, the
original location string is used as a fallback.

Missing locations are not grouped together.

### Actual results

Original postings: 2,468
Diversified representatives: 2,370
Additional postings grouped: 98

Previous representatives: 2,326

Additional opportunities preserved compared
with the earlier location-agnostic approach: 44.

All 2,468 original job IDs were accounted for.

### Engineering decision

Location is part of the duplicate grouping key.

Identical descriptions at different locations
remain separate job recommendations.

The highest-scoring posting is retained as
the representative within each duplicate group.

### Remaining limitation

Near-duplicate descriptions may still appear
as separate recommendations.

These will be investigated separately rather
than automatically merged using fuzzy matching.

Original job records remain unchanged.


## 24. Near-Duplicate Detection — Final Results

Updated:
- app/services/recommendation_diversifier.py
- notebooks/23_apply_recommendation_diversification.py

### Detection methodology

Grouping criteria:
- Normalized company
- Normalized job title
- Normalized location
- Normalized description similarity

Near-duplicate threshold: 0.995 (99.5%)

Similarity implementation:
Python difflib.SequenceMatcher

### Actual results

Original postings: 2,468
Recommendation representatives: 2,341
Additional postings grouped: 127

Original job IDs preserved: 2,468

### SLB investigation

Four SLB Data Scientist postings were investigated.

Their descriptions had similarities between
approximately 99.83% and 100%.

The differences were limited to a minor
scraping artefact in the examined text.

One recommendation representative was retained,
with three alternative job IDs preserved.

### Turing investigation

Repeated Turing job descriptions were also
consolidated within the same normalized location.

### Validation

- All original job IDs accounted for.
- No duplicate representative job IDs.
- Geographic distinctions preserved.
- Alternate job IDs retained.
- Recommendation ranking regenerated successfully.

### Performance limitation

SequenceMatcher can become expensive when
comparing long descriptions.

Future optimization:
Use efficient string similarity calculations,
length-based filtering and candidate blocking
before comparing descriptions.


## Model comparison: TF-IDF vs semantic retrieval

### Evaluation design

Compared three recommendation approaches:

1. TF-IDF lexical retrieval.
2. Sentence Transformer semantic retrieval using all-MiniLM-L6-v2.
3. Semantic retrieval followed by recommendation diversification.

The evaluation pool contained 31 manually judged job postings. Four postings were grouped with existing representative recommendations, producing 27 unique recommendation groups.

Relevance grades:
- 2: Highly relevant.
- 1: Partially relevant.
- 0: Not relevant.

When multiple postings belonged to the same recommendation group, the group received the maximum relevance grade among its reviewed members.

### Group-level evaluation results

| Metric | TF-IDF | Semantic | Diversified Semantic |
|---|---:|---:|---:|
| Precision@5 | 0.6000 | 0.6000 | 0.6000 |
| Precision@10 | 0.4000 | 0.4000 | 0.4000 |
| Recall@10 within judged groups | 0.5714 | 0.5714 | 0.5714 |
| MRR | 1.0000 | 1.0000 | 1.0000 |
| NDCG@10 | 0.7316 | 0.7657 | 0.7657 |

### Interpretation

Semantic retrieval achieved the highest NDCG@10, indicating better ordering of relevant recommendations within this evaluation pool.

All three methods achieved identical precision and recall. Therefore, these results do not demonstrate that semantic retrieval identifies more relevant jobs than TF-IDF.

Diversification produced the same top-10 recommendation groups as semantic retrieval. Its demonstrated benefit is handling duplicate and near-duplicate postings without presenting multiple versions of the same recommendation group.

MRR of 1.0 indicates that the first recommendation was relevant in each evaluated ranking.

Recall@10 of 0.5714 means that 4 of the 7 relevant judged groups appeared in the top 10. This is recall within the manually judged evaluation pool, not recall across the complete job dataset.

### Evaluation limitations

- Only one synthetic candidate profile was evaluated.
- The evaluation contains 31 manually judged postings grouped into 27 recommendation groups.
- Relevance judgments are subjective, although documented guidelines were used.
- Group relevance uses the maximum grade among reviewed group members.
- The evaluation pool was assembled from model-selected results and a random sample; it is not a representative random sample of the complete corpus.
- Results are preliminary offline evidence, not production performance claims.

### Saved evaluation artifacts

- data/processed/model_comparison_metrics.csv
- data/processed/model_comparison_ranked_judgments.csv
- data/processed/model_comparison_group_mapping.csv

## Resume ingestion and candidate profile extraction

### PDF extraction

Implemented `app/services/resume_parser.py` using PyMuPDF.

Responsibilities:
- Validate PDF uploads.
- Enforce a 10 MB file-size limit.
- Enforce a 20-page limit.
- Reject password-protected PDFs.
- Extract text page by page.
- Preserve page boundaries in extracted text.
- Identify PDFs without selectable text and report that OCR is required.

The parser returns structured extraction metadata:
- Extracted text.
- Page count.
- Extracted character count.

Custom `ResumeParsingError` exceptions distinguish expected PDF validation and extraction failures.

Limitation: OCR for scanned resumes has not yet been implemented.

### Candidate profile extraction

Implemented `app/services/candidate_profile.py`.

The initial extraction approach uses transparent, rule-based processing rather than an LLM.

Extracted information:
- Explicitly mentioned technical skills using a maintained skill-alias dictionary.
- Common resume sections.
- Education qualification evidence.
- Target roles supplied separately by the user.

Candidate preferences and target roles are kept separate from qualifications. Target roles are not inferred from resume content, and unmentioned skills are not added.

The resulting `CandidateProfile` contains:
- `skills`
- `education_evidence`
- `sections`
- `target_roles`
- `matching_text`

Limitation: This is an initial rule-based parser. It does not yet reliably calculate employment duration, identify every education format, or understand complex resume layouts.

### Integrated resume pipeline

Implemented `app/services/resume_pipeline.py`.

Processing sequence:

1. Accept PDF bytes.
2. Extract and validate resume text.
3. Build the structured candidate profile.
4. Attach explicitly provided target roles.
5. Return extraction metadata and candidate profile.

The pipeline is reusable by future API endpoints.

### Automated tests

Implemented:
- `tests/test_resume_parser.py`
- `tests/test_candidate_profile.py`
- `tests/test_resume_pipeline.py`

Validation result: 17 tests passed.

Coverage includes valid PDFs, invalid files, empty uploads, file-size limits, page limits, skill extraction, section recognition, education evidence, target-role separation and end-to-end PDF ingestion.

## Reusable job-matching engine

Implemented `app/services/job_matching_engine.py`.

### Architecture

The matching engine consolidates existing research implementations into a reusable application service.

It loads:
- Cleaned job dataset: `jobs_cleaned.parquet`.
- Chunk metadata: `job_chunk_embedding_metadata.parquet`.
- FAISS index: `faiss_job_chunks.index`.
- Sentence Transformer: `sentence-transformers/all-MiniLM-L6-v2`.

It initializes the TF-IDF vectorizer and embedding model once per engine instance, allowing multiple candidate requests to reuse the same resources.

### Retrieval methods

**TF-IDF retrieval**
- Uses the existing scikit-learn TF-IDF configuration.
- Uses title and job-description text.
- Calculates cosine similarity.
- Produces a complete ranking of eligible jobs.

**Semantic retrieval**
- Uses Sentence Transformer embeddings.
- Reuses the existing FAISS index of 10,095 job-description chunks.
- Splits candidate matching text using the existing tokenizer-based chunker.
- Uses 200 content tokens per chunk and 40-token overlap, subject to the model input limit.
- Searches the complete job-chunk index for each candidate chunk.
- Aggregates results using maximum chunk similarity per job.
- Joins retrieved metadata back to the cleaned job dataset.

**Recommendation diversification**
- Reuses `diversify_recommendations()`.
- Uses the existing 0.995 description-similarity threshold.
- Groups duplicate and near-duplicate postings while preserving original job IDs.

### Runtime and validation

The engine automatically selects Apple MPS when available and otherwise supports CPU inference.

Validated locally using the synthetic candidate profile:
- 2,468 eligible jobs.
- 10,095 indexed job chunks.
- 384-dimensional embeddings.
- 2,468 TF-IDF results.
- 2,468 semantic results.
- 2,341 diversified recommendation representatives.
- Successful execution on Apple MPS.

This was an integration smoke test, not a replacement for the earlier manually judged evaluation.

## Backend dependencies

Installed and pinned in `requirements.txt`:

| Dependency | Version | Purpose |
|---|---|---|
| FastAPI | 0.165.0 | HTTP API framework |
| Uvicorn | 0.54.0 | ASGI server |
| python-multipart | 0.0.34 | Multipart PDF uploads |
| HTTPX | 0.28.1 | API testing |

The FastAPI endpoints and application lifecycle are still pending implementation.


## FastAPI backend

### Application implementation

Implemented `app/main.py` using FastAPI.

The API exposes three initial endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Reports application and matching-engine readiness |
| POST | `/upload-resume` | Extracts resume text and returns a structured candidate profile |
| POST | `/match` | Processes a resume and returns TF-IDF, semantic and diversified recommendations |

### Application lifecycle

The FastAPI lifespan context initializes `JobMatchingEngine` once during application startup.

The initialized engine is stored in application state and reused across requests. This avoids loading the Sentence Transformer model, fitting TF-IDF and reading the FAISS index for every request.

If engine initialization fails, the application reports a degraded health state and returns HTTP 503 for matching requests.

The application releases its engine reference during shutdown.

### Upload handling and validation

- Accepts PDF uploads through multipart form data.
- Limits uploaded file size to 10 MB.
- Uses the existing PDF parser for PDF validation and extraction.
- Keeps uploaded resume content in memory; it does not persist uploaded files.
- Accepts target roles as separate user-provided form fields.
- Restricts requested recommendation count to 1–50.
- Returns HTTP 400 for invalid resume content and HTTP 422 for invalid request parameters.

### API testing

Implemented `tests/test_api.py` using FastAPI TestClient.

The matching engine is mocked in endpoint tests so that API behavior can be tested without loading the full ML model or FAISS index.

Initial API test result: 5 passed.

Coverage:
- Health endpoint and startup lifecycle.
- PDF upload and candidate profile response.
- Matching endpoint response structure.
- Invalid PDF handling.
- Request parameter validation.

### Backend dependencies

Pinned in `requirements.txt`:
- FastAPI 0.165.0
- Uvicorn 0.54.0
- python-multipart 0.0.34
- HTTPX 0.28.1

### Current limitations

- API endpoint tests use a mocked matching engine; real model-backed HTTP integration still needs validation.
- Authentication and rate limiting are not implemented.
- OCR for scanned resumes is not implemented.
- The API is not yet deployed.
- The current backend has not yet been load-tested for concurrent requests.


## Real FastAPI and ML integration validation

### Integration test

Implemented `scripts/smoke_test_api.py`.

Unlike the mocked endpoint tests, this script starts the actual FastAPI application and initializes the real JobMatchingEngine.

It validates the complete workflow:

1. Application startup and model initialization.
2. Health endpoint readiness.
3. Actual PDF resume upload.
4. Candidate-profile extraction.
5. Real TF-IDF matching.
6. Real Sentence Transformer candidate embedding.
7. FAISS search across the indexed job chunks.
8. Recommendation diversification.
9. HTTP response serialization.
10. Application shutdown.

### Verified runtime configuration

- Job records: 2,468 eligible jobs.
- FAISS index: 10,095 chunks.
- Embedding dimensions: 384.
- Embedding model: all-MiniLM-L6-v2.
- Inference device: Apple MPS.
- Candidate chunks in integration test: 1.
- Returned recommendations: 5 per retrieval method.

### Real integration result

The following operations returned successful HTTP responses:

- `GET /health`
- `POST /upload-resume`
- `POST /match`

All real API integration checks passed.

The integration test returned five recommendations for each of:
- TF-IDF.
- Semantic retrieval.
- Diversified semantic retrieval.

The test confirms technical integration, not recommendation quality. The candidate profile used in this API test differs from the earlier manually evaluated synthetic query; the earlier evaluation metrics must not be attributed to this new query.

### Runtime observations

- The application successfully initialized the model on Apple MPS.
- Hugging Face unauthenticated-request notices did not prevent model loading.
- PyTorch emitted a macOS multiprocessing notice that did not affect execution.
- A Sentence Transformer deprecation warning was addressed by switching to `get_embedding_dimension()`.

### Remaining backend limitations

- Authentication and rate limiting are not implemented.
- OCR for scanned PDFs is not implemented.
- Concurrent production workloads have not been tested.
- The API has not been deployed.

## Streamlit frontend

### Application implementation

Implemented `app/streamlit_app.py` using Streamlit 1.67.0.

The frontend communicates with the FastAPI backend over HTTP. It does not initialize or load the ML models directly.

Backend URL configuration:
- Environment variable: `JOB_MATCH_API_URL`
- Local default: `http://127.0.0.1:8000`

### Implemented interface features

- PDF resume upload with a 10 MB client-side limit.
- User-supplied target roles, including multiple roles.
- Configurable recommendation count from 5 to 50.
- Candidate skill and education preview.
- Recognition of resume sections.
- Three recommendation views:
  - TF-IDF baseline.
  - Semantic retrieval.
  - Diversified semantic recommendations.
- Similarity scores, seniority, employment type and location.
- Duplicate posting counts and alternate posting IDs.
- Clear notice that the archived job corpus does not represent verified live vacancies.

### Frontend/backend separation

Streamlit sends multipart HTTP requests to FastAPI.

FastAPI remains responsible for PDF processing, candidate profiling and all model inference. The expensive matching engine is loaded by the backend lifecycle and reused across requests.

### Frontend validation

- Python compilation of `app/streamlit_app.py` passed.
- Multiple target roles are sent as repeated multipart form fields.
- API regression tests: 6 passed.
- Complete automated test suite: 23 passed.

### Current frontend limitations

- Browser-based user acceptance testing is pending.
- Job descriptions and external job links are not yet shown in recommendation cards.
- LLM-generated explanations and skill-gap analysis are not implemented.
- No user authentication or persistent candidate history is implemented.

## Candidate-Aware Job Eligibility Integration

### Objective
Prevent irrelevant, overseas, senior-level and over-experienced jobs from reaching the recommendation rankings.

### Implementation
- Integrated `app/services/job_eligibility.py` into `JobMatchingEngine.match()`.
- Eligibility is evaluated using the candidate's explicitly selected target roles, candidate profile evidence, India location constraint and a maximum explicit experience requirement of one year.
- Seniority, mandatory advanced-degree requirements and experience requirements are evaluated before ranking.
- Academic/project equivalency can be applied when a job explicitly accepts equivalent academic and project experience and the candidate profile contains qualifying bachelor's-level education and substantive project evidence.
- TF-IDF and semantic retrieval receive the same candidate-eligible job ID set.
- FAISS chunk hits belonging to excluded jobs are removed before job-level aggregation.
- The engine does not fall back to ranking the unrestricted corpus when no eligible jobs remain.

### Regression and Integration Validation
- Candidate-aware audit regression checks passed for UiPath, FedEx, Innover, four SLB records and ION.
- Corning academic/project equivalency regression passed.
- OmniReach fresher-opportunity regression passed.
- Complete automated test suite: 66 passed.
- Real FastAPI smoke test passed:
  - Sentence Transformer: `all-MiniLM-L6-v2`
  - Embedding dimensions: 384
  - FAISS indexed chunks: 10,095
  - Runtime device: Apple MPS
  - Resume upload endpoint: HTTP 200
  - Matching endpoint: HTTP 200
  - TF-IDF, semantic and diversified result lists returned successfully.

### Observed Recommendation Results
For the smoke-test candidate targeting Junior Data Analyst, the top results included Data Analyst Intern roles at RegisterKaro and Proziod Analytics, a Data Analyst role at Mantle Solutions, an Online Data Analyst role at TELUS Digital and a Telecom Data Analyst role at TEOCO.

These results demonstrate improved role-family alignment after candidate-aware eligibility filtering. They do not establish that every recommendation satisfies all candidate-specific requirements.

### Limitations
- The current job corpus is archived data, not a live vacancy feed.
- Experience and seniority detection use rules and may produce false positives or false negatives.
- Title and eligibility filtering do not guarantee that every hard skill, language, schedule or other job-specific requirement is satisfied.
- The smoke test uses a test resume; the user's personal resume still requires a separate end-to-end review.
- Recommendation quality must continue to be assessed through manual relevance judgments and offline evaluation.

## Eligibility Metrics: API and Streamlit

### FastAPI Response
- Added `eligible_job_count` and `excluded_job_count` to `MatchResponse`.
- `/match` returns both values from the candidate-aware matching engine.
- Added API regression assertions using a mocked engine with known fixture values.
- API test suite: 6 passed.

### Streamlit Interface
- Added Eligible jobs and Excluded jobs metrics above the recommendation tabs.
- Added a short explanation of the eligibility criteria.
- Existing recommendation tabs and job cards remain unchanged.

### Real Integration Validation
- Updated `scripts/smoke_test_api.py` to validate both counters.
- Verified that both values are integers, at least one job is eligible, and eligible plus excluded equals the source corpus size.
- Test candidate target role: Junior Data Analyst.
- Source corpus: 2,468 jobs.
- Eligible jobs: 5.
- Excluded jobs: 2,463.
- TF-IDF, semantic and diversified ranking each returned five results.
- Real FastAPI integration checks passed using Sentence Transformers, FAISS and Apple MPS.

The eligibility count is candidate- and target-role-specific. It is not a general count of all potentially suitable jobs. The source dataset remains archived and does not represent verified live vacancies.

## Resume Education Extraction Correction

### Problem
The previous education extractor searched the entire resume. Degree mentions in professional summaries could therefore be incorrectly classified as formal education evidence. Comma-based truncation also removed institution, dates and grades.

### Correction
- Education evidence is now extracted exclusively from the recognized Education section.
- Complete qualification lines are preserved, including degree specialization, institution, dates, CGPA and marks.
- Separate certification clauses are excluded from degree evidence.
- A degree mentioned only in the professional summary is not treated as formal education evidence.

### Regression Validation
- Added a test for complete B.Tech and Diploma entries.
- Added a test preventing summary-to-education contamination.
- Existing certification cleanup and API education tests pass.
- Complete test suite: 68 passed.

### Remaining Limitation
Education extraction remains rule-based and depends on recognizable section headings and selectable PDF text. Scanned resumes and unusual layouts may require OCR or additional parsing rules.

## De-identified Candidate Resume End-to-End Validation

### Resume Ingestion

- Tested with a real one-page candidate PDF.
- Extracted 3,430 characters.
- Detected 11 explicitly mentioned skills.
- Successfully extracted two education entries.
- Institution names, qualification dates and grades are intentionally omitted from this public engineering record.
- Education evidence no longer includes professional-summary text.

### Matching

- User-selected target role: Junior Data Analyst.
- Eligible postings: 5.
- Excluded postings: 2,463.
- Candidate chunks: 5.
- Indexed job chunks: 10,095.
- Runtime device: Apple MPS.
- Diversified, semantic and TF-IDF results rendered successfully.

### Observations

- Five recommendations were returned across data analyst, data analyst intern and telecom data analyst categories.
- The engine honors explicitly selected target roles rather than inferring career preferences from unrelated resume sections.
- Language requirements are not currently validated by the eligibility filter and require manual review.

### Limitations

- All recommendations originate from the archived dataset and are not verified live vacancies.
- Candidate-specific resume content is not included in the repository.

---

## 25. GitHub Reproducibility and Cross-Platform Portability
Date: 2026-11-24

### Hardware portability

Created:
- `app/services/device_utils.py`
- `tests/test_device_utils.py`

Implemented a shared hardware selection policy:
1. CUDA when available.
2. Apple MPS when CUDA is unavailable.
3. CPU fallback.

Integrated the selector into:
- `app/services/job_matching_engine.py`
- `notebooks/19_generate_embeddings.py`

Four automated tests cover the device-selection behavior.

The application has been exercised locally on Apple MPS. CUDA and CPU hardware have not been independently benchmarked.

### Dependency management

Created:
- `requirements.txt`
- `requirements-dev.txt`
- `.python-version`

The dependency manifests declare pinned direct dependencies, including:
- FastAPI and Uvicorn.
- Streamlit.
- NumPy and Pandas.
- PyArrow and DuckDB.
- Scikit-learn.
- PyTorch and Sentence Transformers.
- Transformers and Hugging Face Hub.
- `faiss-cpu`.
- PyMuPDF.
- Pytest for development and testing.

The tested Python version is 3.14.3.

The existing environment passed `pip check` with no broken requirements. A separate clean-machine installation has not yet been independently verified.

### Reference-data organization

Moved:
`data/processed/location_aliases.csv`

To:
`data/reference/location_aliases.csv`

Updated `notebooks/05_normalize_locations.py` to use the tracked reference-data path.

Raw source data and generated processed artifacts remain excluded from Git.

### Reproducible data preparation

Created `scripts/prepare_data.py`.

Validation-only command:
`python scripts/prepare_data.py`

Explicit pipeline execution:
`python scripts/prepare_data.py --run`

The seven-step pipeline performs:
1. Location normalization.
2. Job dataset cleaning.
3. Token-length auditing.
4. Chunk-volume estimation.
5. Job chunk generation.
6. Embedding generation.
7. FAISS index construction.

Validation passed locally and confirmed the required inputs and scripts.

The full rebuild was not executed because the existing processed datasets, embeddings and FAISS index were already available. This preserved the working artifacts.

### Local environment configuration

Created `.env.example` with defaults for:
- Ollama base URL.
- Qwen3 4B model.
- Inference context and output limits.
- Inference timeout.
- FastAPI URL used by Streamlit.

The application reads environment variables directly. The example file must be copied and sourced by the developer; Python does not automatically load it.

### Documentation

Updated the project README with the implemented architecture, installation steps, dataset preparation, local inference configuration, API/frontend startup, evaluation evidence and limitations.

Created `docs/project-walkthrough.md` to explain the technical architecture and engineering decisions.

The existing engineering history and relevance-judgment guidelines were retained.

### Final regression verification

Date: 2026-11-24

- Python syntax compilation: passed.
- Data preparation validation: passed.
- `pip check`: no broken requirements.
- Pytest: 92 passed.
- Test failures: 0.
- Execution time: 8.01 seconds.
- Warnings: 5 non-blocking SWIG/PyMuPDF deprecation warnings.

### Remaining GitHub work

- Inspect all files intended for the first commit.
- Verify that private resumes, raw job data, generated artifacts and secrets are excluded.
- Check the final Git diff and whitespace.
- Create the first implementation commit.
- Publish the repository after reviewing dataset handling and licensing.

No live job ingestion, OCR, authentication, production deployment or load testing is claimed as implemented.


---

## 26. Recruiter Demonstration Scope
Date: 2026-11-24

### Scope decision
The portfolio milestone now uses a verified local demonstration rather than a continuously hosted public AI service. This avoids introducing ongoing compute cost, model-serving operations and public resume-upload exposure before those capabilities are necessary.

### Demo assets prepared
- Added `data/sample/demo_candidate_resume.txt`, containing a clearly synthetic candidate profile.
- Added `scripts/create_demo_resume.py` to generate and re-open a one-page PDF for the demo.
- Added `docs/demo-recording-guide.md` with local startup checks, real API smoke-test steps, manual acceptance criteria, screenshot capture plan and a short video outline.
- Updated the README and technical walkthrough to explain the recorded-demo approach.

### Validation status
The generator is source-reviewed but has not yet been executed in the user's local Python environment. The real API smoke test, manual Streamlit workflow, screenshot capture and video recording remain pending local execution. No public demo assets are claimed as captured.
