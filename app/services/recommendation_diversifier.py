import hashlib
import html
import re
from difflib import SequenceMatcher

import pandas as pd


def normalize_description(text):
    text = html.unescape(str(text))
    text = text.replace("<![CDATA[", "")
    text = text.replace("]]>", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_label(text):
    text = html.unescape(str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_location_key(row):
    city = row.get("normalized_city")
    state = row.get("normalized_state")
    country = row.get("normalized_country")

    if pd.notna(city) and pd.notna(country):
        return "|".join(
            normalize_label(str(value))
            for value in [city, state, country]
            if pd.notna(value)
        )

    location = row.get("location")

    if pd.isna(location) or not str(location).strip():
        return f"unknown:{row['job_id']}"

    return f"raw:{normalize_label(str(location))}"


def diversify_recommendations(
    ranked_jobs: pd.DataFrame,
    similarity_threshold: float = 0.995
) -> pd.DataFrame:

    required = {
        "job_id", "title", "company", "location",
        "description", "semantic_similarity", "semantic_rank"
    }

    missing = required - set(ranked_jobs.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not 0 < similarity_threshold <= 1:
        raise ValueError("Invalid similarity threshold.")

    jobs = ranked_jobs.copy()

    if jobs.empty:
        return jobs

    if jobs["job_id"].duplicated().any():
        raise ValueError("Duplicate job IDs in input.")

    if jobs["description"].isna().any():
        raise ValueError("Missing job descriptions.")

    jobs["_company"] = jobs["company"].map(normalize_label)
    jobs["_title"] = jobs["title"].map(normalize_label)

    jobs["_description"] = jobs["description"].map(
        normalize_description
    )

    jobs["_location"] = jobs.apply(
        build_location_key,
        axis=1
    )

    group_columns = [
        "_company", "_title", "_location"
    ]

    jobs = jobs.sort_values(
        ["semantic_similarity", "semantic_rank"],
        ascending=[False, True],
        kind="stable"
    )

    output = []

    # Compare descriptions only within the same
    # company, title and normalized location.
    for _, group in jobs.groupby(group_columns, sort=False):

        clusters = []

        for _, row in group.iterrows():

            description = row["_description"]

            best_cluster = None
            best_similarity = -1.0

            # Match against existing cluster representatives.
            for cluster in clusters:

                similarity = SequenceMatcher(
                    None,
                    cluster["description"],
                    description,
                    autojunk=False
                ).ratio()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if (
                best_cluster is not None
                and best_similarity >= similarity_threshold
                and description
            ):
                best_cluster["members"].append(row)
                best_cluster["similarities"].append(best_similarity)

            else:
                clusters.append({
                    "description": description,
                    "members": [row],
                    "similarities": []
                })

        # Keep the highest-scoring representative.
        for cluster in clusters:

            members = cluster["members"]
            representative = members[0].to_dict()

            member_ids = [
                str(member["job_id"])
                for member in members
            ]

            if len(members) == 1:
                match_type = "single"
            elif all(
                member["_description"] == members[0]["_description"]
                for member in members
            ):
                match_type = "exact"
            else:
                match_type = "near_duplicate"

            representative["duplicate_posting_count"] = len(members)

            representative["alternate_job_ids"] = ", ".join(
                member_ids[1:]
            )

            representative["duplicate_match_type"] = match_type

            representative["minimum_description_similarity"] = (
                min(cluster["similarities"])
                if cluster["similarities"]
                else 1.0
            )

            output.append(representative)

    result = pd.DataFrame(output)

    internal_columns = [
        "_company",
        "_title",
        "_location",
        "_description"
    ]

    result = result.drop(
        columns=internal_columns,
        errors="ignore"
    )

    result = result.sort_values(
        ["semantic_similarity", "semantic_rank"],
        ascending=[False, True],
        kind="stable"
    ).reset_index(drop=True)

    result.insert(
        0,
        "diversified_rank",
        range(1, len(result) + 1)
    )

    return result
