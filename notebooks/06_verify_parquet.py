from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs_with_normalized_locations.parquet"
)


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("\nPARQUET DATASET VALIDATION")
    print("=" * 55)

    print(
        "File size:",
        round(DATA_PATH.stat().st_size / (1024 * 1024), 2),
        "MB"
    )

    connection = duckdb.connect(database=":memory:")

    # Count records and unique job IDs directly from Parquet
    row_count, unique_ids = connection.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT id) AS unique_job_ids
        FROM read_parquet(?)
        """,
        [str(DATA_PATH)]
    ).fetchone()

    print("\n1. RECORD VALIDATION")
    print("Total rows:", row_count)
    print("Unique job IDs:", unique_ids)
    print("Duplicate job IDs:", row_count - unique_ids)

    # Check geographic classification
    print("\n2. NORMALIZATION STATUS")

    status = connection.execute(
        """
        SELECT
            normalization_method,
            COUNT(*) AS job_count
        FROM read_parquet(?)
        GROUP BY normalization_method
        ORDER BY job_count DESC
        """,
        [str(DATA_PATH)]
    ).fetchdf()

    print(status.to_string(index=False))

    # Check country distribution
    print("\n3. COUNTRY DISTRIBUTION")

    countries = connection.execute(
        """
        SELECT
            COALESCE(normalized_country, 'Unclassified') AS country,
            COUNT(*) AS job_count
        FROM read_parquet(?)
        GROUP BY normalized_country
        ORDER BY job_count DESC
        """,
        [str(DATA_PATH)]
    ).fetchdf()

    print(countries.to_string(index=False))

    # Verify that no records were accidentally lost
    if row_count != 2500:
        raise ValueError("Record count does not match the original dataset!")

    print("\nVALIDATION PASSED")
    print("All 2,500 original job records are preserved.")

    connection.close()


if __name__ == "__main__":
    main()
