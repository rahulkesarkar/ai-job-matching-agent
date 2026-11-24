from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_cleaned.parquet"

CANDIDATE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_candidate_profile.txt"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tfidf_ranked_jobs.csv"


def main():

    print("\nAI JOB MATCHING AGENT - TF-IDF BASELINE")
    print("=" * 60)

    # 1. Load cleaned job data
    jobs = pd.read_parquet(DATA_PATH)

    eligible_jobs = jobs[
        jobs["eligible_for_semantic_matching"].eq(True)
    ].copy()

    print("Eligible jobs loaded:", len(eligible_jobs))

    # 2. Load candidate profile
    candidate_raw = CANDIDATE_PATH.read_text(
        encoding="utf-8"
    )

    # Exclude geographic and employment preferences
    candidate_text = candidate_raw.split(
        "Target Roles:", 1
    )[1].split("Career Preferences:", 1)[0]

    candidate_text = "Target Roles: " + candidate_text.strip()

    if not candidate_text.strip():
        raise ValueError("Candidate profile is empty.")

    # 3. Prepare job documents
    job_documents = (
        eligible_jobs["title"].fillna("").astype(str)
        + " "
        + eligible_jobs["description"].fillna("").astype(str)
    ).tolist()

    # 4. Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    # Learn vocabulary and IDF weights from job descriptions
    job_matrix = vectorizer.fit_transform(job_documents)

    # 5. Convert candidate profile using the same vocabulary
    candidate_vector = vectorizer.transform(
        [candidate_text]
    )

    # 6. Calculate cosine similarity
    similarity_scores = cosine_similarity(
        candidate_vector,
        job_matrix
    ).flatten()

    # 7. Rank jobs by similarity
    ranked_jobs = eligible_jobs.copy()

    ranked_jobs["tfidf_similarity"] = similarity_scores

    ranked_jobs = ranked_jobs.sort_values(
        by="tfidf_similarity",
        ascending=False
    )

    # 8. Save complete ranking
    result_columns = [
        "id",
        "title",
        "company",
        "location",
        "seniority_level",
        "employment_type",
        "tfidf_similarity"
    ]

    ranked_jobs[result_columns].to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 9. Display top 10 recommendations
    print("\nTOP 10 JOB MATCHES")
    print("=" * 60)

    print(
        ranked_jobs[result_columns]
        .head(10)
        .round({"tfidf_similarity": 4})
        .to_string(index=False)
    )

    print("\nMODEL INFORMATION")
    print("-" * 60)

    print("Vectorizer: TF-IDF")
    print("Similarity metric: Cosine Similarity")
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Jobs compared:", len(eligible_jobs))

    print("\nIMPORTANT:")
    print("Similarity scores are not match probabilities.")

    print("\nComplete ranking saved to:")
    print(OUTPUT_PATH)

    print("\nTF-IDF MATCHING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
