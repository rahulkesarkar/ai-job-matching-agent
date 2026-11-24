from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.candidate_profile import CandidateProfile
from app.services.job_eligibility import (
    JobEligibilityError,
    filter_jobs_for_candidate,
)
from app.services.recommendation_diversifier import (
    diversify_recommendations,
)
from app.services.text_chunker import chunk_text
from app.services.device_utils import select_torch_device


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSIONS = 384


@dataclass
class JobMatchResults:
    tfidf_ranked: pd.DataFrame
    semantic_ranked: pd.DataFrame
    diversified_ranked: pd.DataFrame
    candidate_chunk_count: int
    indexed_chunk_count: int
    model_device: str
    eligible_job_count: int = 0
    excluded_job_count: int = 0


class JobMatchingError(RuntimeError):
    """Raised when the matching engine cannot complete retrieval."""


class JobMatchingEngine:
    """
    Reusable TF-IDF and semantic job-matching engine.

    Heavy resources are loaded once when the engine is initialized.
    Individual candidate profiles can then be matched repeatedly.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        device: str | None = None,
    ):
        self.project_root = (
            Path(project_root)
            if project_root
            else Path(__file__).resolve().parents[2]
        )

        self.data_dir = self.project_root / "data" / "processed"

        self.jobs_path = self.data_dir / "jobs_cleaned.parquet"
        self.metadata_path = (
            self.data_dir / "job_chunk_embedding_metadata.parquet"
        )
        self.index_path = self.data_dir / "faiss_job_chunks.index"

        # Use the shared cross-platform hardware selection policy.
        if device is None:
            device = select_torch_device()

        self.device = device

        # Load and validate job data.
        self.jobs = pd.read_parquet(self.jobs_path)

        required_job_columns = {
            "id",
            "title",
            "description",
            "company",
            "location",
            "eligible_for_semantic_matching",
        }

        missing = required_job_columns - set(self.jobs.columns)

        if missing:
            raise JobMatchingError(
                f"Job dataset is missing columns: {sorted(missing)}"
            )

        if self.jobs["id"].duplicated().any():
            raise JobMatchingError("Duplicate job IDs in cleaned dataset.")

        self.eligible_jobs = self.jobs[
            self.jobs["eligible_for_semantic_matching"].eq(True)
        ].copy()

        self.jobs["id"] = self.jobs["id"].astype(str)
        self.eligible_jobs["id"] = (
            self.eligible_jobs["id"].astype(str)
        )

        if self.eligible_jobs.empty:
            raise JobMatchingError(
                "No jobs are eligible for matching."
            )

        # Fit TF-IDF once using the job corpus.
        job_documents = (
            self.eligible_jobs["title"].fillna("").astype(str)
            + " "
            + self.eligible_jobs["description"].fillna("").astype(str)
        ).tolist()

        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )

        self.job_tfidf_matrix = (
            self.tfidf_vectorizer.fit_transform(job_documents)
        )

        # Load chunk metadata and the previously built FAISS index.
        self.metadata = pd.read_parquet(self.metadata_path)
        self.metadata["job_id"] = (
            self.metadata["job_id"].astype(str)
        )

        self.index = faiss.read_index(str(self.index_path))

        if self.index.ntotal != len(self.metadata):
            raise JobMatchingError(
                "FAISS index count does not match chunk metadata."
            )

        if self.index.ntotal == 0:
            raise JobMatchingError("FAISS index is empty.")

        if self.index.d != EXPECTED_DIMENSIONS:
            raise JobMatchingError(
                "Unexpected FAISS embedding dimensions."
            )

        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise JobMatchingError(
                "Expected an inner-product FAISS index."
            )

        if not self.metadata["chunk_id"].is_unique:
            raise JobMatchingError(
                "Duplicate chunk IDs found in metadata."
            )

        indexed_job_ids = set(self.metadata["job_id"])
        available_job_ids = set(self.eligible_jobs["id"])

        if not indexed_job_ids.issubset(available_job_ids):
            raise JobMatchingError(
                "Some indexed job IDs are missing from the cleaned dataset."
            )

        # Load the embedding model once.
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device,
        )

        if (
            self.model.get_embedding_dimension()
            != EXPECTED_DIMENSIONS
        ):
            raise JobMatchingError(
                "Embedding model dimensions do not match FAISS index."
            )

        self.max_seq_length = int(self.model.max_seq_length)

        print("Job Matching Engine initialized")
        print("Eligible jobs:", len(self.eligible_jobs))
        print("Indexed jobs:", len(indexed_job_ids))
        print("Indexed chunks:", self.index.ntotal)
        print("Embedding dimensions:", self.index.d)
        print("Model device:", self.model.device)

    def _validate_candidate(self, profile: CandidateProfile) -> str:
        """Validate and return candidate matching text."""
        matching_text = getattr(profile, "matching_text", "")

        if not isinstance(matching_text, str) or not matching_text.strip():
            raise JobMatchingError(
                "Candidate matching text cannot be empty."
            )

        return matching_text.strip()

    def rank_tfidf(
        self,
        candidate_text: str,
        allowed_job_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Rank eligible jobs using TF-IDF cosine similarity."""
        candidate_vector = self.tfidf_vectorizer.transform(
            [candidate_text]
        )

        scores = cosine_similarity(
            candidate_vector,
            self.job_tfidf_matrix,
        ).flatten()

        ranked = self.eligible_jobs.copy()
        ranked["tfidf_similarity"] = scores

        if allowed_job_ids is not None:
            ranked = ranked[
                ranked["id"].isin(allowed_job_ids)
            ].copy()

        if ranked.empty:
            raise JobMatchingError(
                "No eligible jobs are available for TF-IDF ranking."
            )

        ranked = ranked.sort_values(
            "tfidf_similarity",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)

        ranked.insert(
            0,
            "tfidf_rank",
            range(1, len(ranked) + 1),
        )

        return ranked

    def rank_semantic(
        self,
        candidate_text: str,
        allowed_job_ids: set[str] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """
        Rank jobs using candidate and job chunk embeddings.

        Each candidate chunk searches the complete FAISS index.
        The maximum similarity across candidate/job chunk pairs
        becomes the score for that job.
        """
        tokenizer = self.model.tokenizer

        special_tokens = tokenizer.num_special_tokens_to_add(
            pair=False
        )

        chunk_size = min(
            200,
            self.max_seq_length - special_tokens,
        )

        overlap = min(40, chunk_size - 1)

        candidate_chunks = chunk_text(
            text=candidate_text,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not candidate_chunks:
            raise JobMatchingError(
                "No usable candidate text chunks were generated."
            )

        candidate_embeddings = self.model.encode(
            [chunk["text"] for chunk in candidate_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        candidate_embeddings = np.ascontiguousarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        expected_shape = (
            len(candidate_chunks),
            EXPECTED_DIMENSIONS,
        )

        if candidate_embeddings.shape != expected_shape:
            raise JobMatchingError(
                "Candidate embedding shape is incorrect."
            )

        if not np.isfinite(candidate_embeddings).all():
            raise JobMatchingError(
                "Candidate embeddings contain non-finite values."
            )

        norms = np.linalg.norm(
            candidate_embeddings,
            axis=1,
        )

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise JobMatchingError(
                "Candidate embeddings are not normalized."
            )

        # Search every job chunk for every candidate chunk.
        scores, indices = self.index.search(
            candidate_embeddings,
            self.index.ntotal,
        )

        valid_mask = indices >= 0

        vector_ids = indices[valid_mask]
        similarity_scores = scores[valid_mask]

        if len(vector_ids) != (
            len(candidate_chunks) * self.index.ntotal
        ):
            raise JobMatchingError(
                "FAISS search did not return every indexed chunk."
            )

        # Map returned FAISS vector IDs to their metadata.
        hits = self.metadata.iloc[vector_ids].copy()
        hits = hits.reset_index(drop=True)

        hits["semantic_similarity"] = similarity_scores

        # Remove chunks belonging to jobs rejected by eligibility rules.
        if allowed_job_ids is not None:
            hits = hits[
                hits["job_id"].isin(allowed_job_ids)
            ].copy()

        if hits.empty:
            raise JobMatchingError(
                "FAISS returned no chunks for eligible jobs."
            )

        # Keep the best candidate/job chunk-pair score per job.
        best_indices = (
            hits.groupby("job_id")["semantic_similarity"].idxmax()
        )

        best_hits = hits.loc[
            best_indices,
            [
                "job_id",
                "semantic_similarity",
                "chunk_id",
                "chunk_index",
            ],
        ].copy()

        # Recover full job descriptions and metadata.
        ranked = best_hits.merge(
            self.eligible_jobs,
            left_on="job_id",
            right_on="id",
            how="left",
            validate="one_to_one",
        )

        if ranked["description"].isna().any():
            raise JobMatchingError(
                "Some semantic results could not be joined to job data."
            )

        ranked = ranked.sort_values(
            "semantic_similarity",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)

        if ranked["job_id"].duplicated().any():
            raise JobMatchingError(
                "Duplicate jobs remain after semantic aggregation."
            )

        ranked.insert(
            0,
            "semantic_rank",
            range(1, len(ranked) + 1),
        )

        return ranked, len(candidate_chunks)

    def match(
        self,
        profile: CandidateProfile,
    ) -> JobMatchResults:
        """
        Filter jobs for candidate eligibility, then produce three rankings.

        Eligibility is evaluated before TF-IDF and semantic ranking.
        Both retrieval methods receive exactly the same allowed job IDs.
        """
        try:
            candidate_text = self._validate_candidate(profile)

            eligibility = filter_jobs_for_candidate(
                jobs=self.eligible_jobs,
                target_roles=profile.target_roles,
                max_required_experience_years=1,
                preferred_country="India",
                candidate_profile=profile,
            )

            allowed_job_ids = set(
                eligibility.eligible_jobs["id"].astype(str)
            )

            if not allowed_job_ids:
                raise JobMatchingError(
                    "No jobs passed the current eligibility filters. "
                    "Try reviewing your target roles or experience settings."
                )

            tfidf_ranked = self.rank_tfidf(
                candidate_text,
                allowed_job_ids=allowed_job_ids,
            )

            semantic_ranked, candidate_chunk_count = (
                self.rank_semantic(
                    candidate_text,
                    allowed_job_ids=allowed_job_ids,
                )
            )

            diversified_ranked = diversify_recommendations(
                semantic_ranked,
                similarity_threshold=0.995,
            )

            return JobMatchResults(
                tfidf_ranked=tfidf_ranked,
                semantic_ranked=semantic_ranked,
                diversified_ranked=diversified_ranked,
                candidate_chunk_count=candidate_chunk_count,
                indexed_chunk_count=self.index.ntotal,
                model_device=str(self.model.device),
                eligible_job_count=eligibility.eligible_count,
                excluded_job_count=eligibility.excluded_count,
            )

        except (JobMatchingError, JobEligibilityError) as exc:
            if isinstance(exc, JobMatchingError):
                raise
            raise JobMatchingError(str(exc)) from exc

        except Exception as exc:
            raise JobMatchingError(
                "Unable to generate job recommendations."
            ) from exc
