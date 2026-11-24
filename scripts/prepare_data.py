"""Prepare job data, embeddings and the FAISS search index."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

RAW_DATA = ROOT / "data" / "raw" / "role_radar_scraped_jobs.json"
LOCATION_ALIASES = ROOT / "data" / "reference" / "location_aliases.csv"

STEPS = [
    (
        "Normalize job locations",
        "notebooks/05_normalize_locations.py",
        ["jobs_with_normalized_locations.parquet"],
    ),
    (
        "Clean job dataset",
        "notebooks/07_clean_dataset.py",
        ["jobs_cleaned.parquet"],
    ),
    (
        "Audit token lengths",
        "notebooks/14_audit_token_lengths.py",
        ["token_length_audit.csv"],
    ),
    (
        "Estimate chunk volume",
        "notebooks/16_estimate_chunk_volume.py",
        ["chunk_volume_estimate.csv"],
    ),
    (
        "Generate job chunks",
        "notebooks/17_generate_job_chunks.py",
        ["job_chunks.parquet"],
    ),
    (
        "Generate embeddings",
        "notebooks/19_generate_embeddings.py",
        [
            "job_chunk_embeddings.npy",
            "job_chunk_embedding_metadata.parquet",
            "embedding_manifest.json",
        ],
    ),
    (
        "Build FAISS index",
        "notebooks/20_build_faiss_index.py",
        [
            "faiss_job_chunks.index",
            "faiss_index_manifest.json",
        ],
    ),
]


def validate_inputs() -> None:
    """Verify required source files and preparation scripts exist."""

    required_files = [RAW_DATA, LOCATION_ALIASES]

    missing = [path for path in required_files if not path.is_file()]

    for _, script, _ in STEPS:
        script_path = ROOT / script
        if not script_path.is_file():
            missing.append(script_path)

    if missing:
        details = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required files are missing:\n{details}"
        )


def run_pipeline() -> None:
    """Run the complete data and search-index preparation pipeline."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(ROOT)
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )

    for number, (label, script, outputs) in enumerate(STEPS, start=1):
        print(
            f"\n[{number}/{len(STEPS)}] {label}",
            flush=True,
        )

        subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            env=env,
            check=True,
        )

        missing_outputs = [
            name for name in outputs
            if not (DATA_DIR / name).is_file()
        ]

        if missing_outputs:
            raise RuntimeError(
                f"{label} did not produce: {', '.join(missing_outputs)}"
            )

    print("\nAll data and FAISS index build steps completed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare job data and build the semantic search index."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the full pipeline. Existing processed artifacts may be overwritten.",
    )
    args = parser.parse_args()

    validate_inputs()

    print("Required input files and preparation scripts are present.")

    for number, (label, script, _) in enumerate(STEPS, start=1):
        print(f"{number}. {label} ({script})")

    if not args.run:
        print(
            "\nValidation only. Use --run to execute the pipeline."
        )
        return

    run_pipeline()


if __name__ == "__main__":
    main()
