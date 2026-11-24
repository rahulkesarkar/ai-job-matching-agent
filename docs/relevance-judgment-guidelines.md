# Job Matching — Relevance Evaluation Guidelines

## Purpose

Evaluate whether the job-matching system retrieves
jobs that are genuinely suitable for a candidate.

The initial evaluation uses a synthetic candidate
profile and human-reviewed job descriptions.

## Candidate Profile

Target roles:
- Data Analyst
- Data Scientist
- Machine Learning Engineer
- AI Engineer
- Data Engineer
- Related entry-level technical roles

Skills:
- Python
- SQL
- Pandas
- NumPy
- Scikit-learn
- Data Analysis
- Data Cleaning
- Exploratory Data Analysis
- Machine Learning
- Feature Engineering
- Model Evaluation
- Git and GitHub

Preferences:
- Entry-level opportunities
- Full-time employment
- Locations across India

## Relevance Labels

### Grade 2 — Highly Relevant

The job:
- Belongs to a suitable technical role.
- Has meaningful overlap with candidate skills.
- Accepts freshers, entry-level or early-career candidates.
- Does not contain a clear disqualifying requirement.

### Grade 1 — Partially Relevant

The job:
- Has meaningful technical or skill overlap.
- Has uncertain or conflicting seniority information.
- Is related to the candidate's target roles but
  does not fully match.
- May have a preference mismatch without a clear
  disqualifying requirement.

### Grade 0 — Not Relevant

The job:
- Belongs to an unrelated role.
- Requires substantial experience the candidate
  does not have.
- Has a mandatory qualification the candidate
  does not meet.
- Has a major location or employment mismatch.

## Judgment Rules

1. Read the actual job description, not just the title.
2. Use explicit experience requirements as evidence.
3. Do not assume Associate means entry-level.
4. Do not assume Mid-Senior means experienced if
   the description explicitly accepts freshers.
5. Treat job metadata and description conflicts
   as something to investigate.
6. Do not invent missing candidate qualifications.
7. Record a short reason for every judgment.
8. Do not use TF-IDF scores to decide relevance.

## Evaluation Scope

The first annotation round will examine the top
10 TF-IDF results.

This is a pilot evaluation, not a complete estimate
of retrieval performance across the entire dataset.

A larger evaluation pool will be created later using
the results from multiple retrieval approaches.

## Planned Metrics

Precision@K:
Fraction of retrieved jobs that are relevant.

Recall@K:
Fraction of known relevant jobs retrieved in the
defined evaluation pool.

MRR:
Measures how highly the first relevant result ranks.

NDCG@K:
Measures ranking quality while accounting for
graded relevance.

## Important Limitations

- The initial candidate is synthetic.
- Relevance judgments are human-assigned.
- The first evaluation pool is incomplete.
- Recall must not be reported as full-dataset recall
  when unjudged jobs remain.
- Evaluation labels must remain independent
  of model-generated similarity scores.

