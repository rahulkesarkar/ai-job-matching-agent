import sys
from pathlib import Path

# Add the project root so the app package is importable
# when this script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf
from fastapi.testclient import TestClient

from app.main import app


def create_test_resume():
    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        """
        PROFESSIONAL SUMMARY
        AI and Machine Learning graduate with Python project experience.

        TECHNICAL SKILLS
        Python, SQL, Pandas, NumPy, Scikit-learn, Git.

        EDUCATION
        B.Tech in Artificial Intelligence and Machine Learning.

        PROJECTS
        Developed data analysis and machine learning projects using
        Python, SQL, data cleaning and exploratory data analysis.
        """
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def main():
    print("\nREAL FASTAPI INTEGRATION TEST")
    print("=" * 55)

    with TestClient(app) as client:
        # 1. Verify real application startup
        health = client.get("/health")
        assert health.status_code == 200, health.text

        health_data = health.json()
        print("\nHEALTH CHECK:", health_data)

        assert health_data["matching_engine_ready"] is True

        # 2. Test real PDF upload and profile extraction
        resume = create_test_resume()

        upload = client.post(
            "/upload-resume",
            files={
                "file": (
                    "integration_resume.pdf",
                    resume,
                    "application/pdf",
                )
            },
            data={
                "target_roles": "Junior Data Analyst",
            },
        )

        assert upload.status_code == 200, upload.text

        upload_data = upload.json()

        print("\nRESUME UPLOAD: PASSED")
        print("Skills:", upload_data["skills"])
        print("Target roles:", upload_data["target_roles"])

        # 3. Test real ML matching through HTTP
        match = client.post(
            "/match",
            files={
                "file": (
                    "integration_resume.pdf",
                    resume,
                    "application/pdf",
                )
            },
            data={
                "target_roles": "Junior Data Analyst",
                "top_k": "5",
            },
        )

        assert match.status_code == 200, match.text

        match_data = match.json()
        results = match_data["results"]

        print("\nREAL MATCHING: PASSED")
        print("Model device:", match_data["model_device"])
        print("Candidate chunks:", match_data["candidate_chunks"])
        print("Indexed chunks:", match_data["indexed_chunks"])

        # 4. Validate eligibility counters from the real matching engine.
        eligible_count = match_data["eligible_job_count"]
        excluded_count = match_data["excluded_job_count"]

        assert isinstance(eligible_count, int)
        assert isinstance(excluded_count, int)
        assert eligible_count > 0
        assert excluded_count >= 0

        source_job_count = len(
            app.state.matching_engine.eligible_jobs
        )

        assert eligible_count + excluded_count == source_job_count

        print("Eligible jobs:", eligible_count)
        print("Excluded jobs:", excluded_count)
        print("Source corpus jobs:", source_job_count)
        print("ELIGIBILITY COUNTERS: PASSED")

        for method, recommendations in results.items():
            print(f"\n{method.upper()} TOP RESULTS")

            assert len(recommendations) == 5

            for rank, job in enumerate(recommendations, start=1):
                print(
                    f"{rank}. {job['title']} | "
                    f"{job['company']}"
                )

        assert "Python" in match_data["candidate_skills"]

        print("\nALL REAL API INTEGRATION CHECKS PASSED")
        print("=" * 55)


if __name__ == "__main__":
    main()
