import sys
from pathlib import Path

# Add the project root so the app package is importable
# when this script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf

from fastapi.testclient import TestClient

from app.main import app
from app.services.job_eligibility import filter_jobs_for_candidate
from app.services.resume_pipeline import process_resume_pdf


def create_test_resume():
    """Create a synthetic resume with explicit evidence for all test requirements."""
    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        """
        PROFESSIONAL SUMMARY

        AI and Machine Learning graduate with data analysis project experience.

        TECHNICAL SKILLS

        Python, SQL, Power BI, Tableau, Data Analysis,
        Data Visualization, Excel, Pandas, NumPy,
        Scikit-learn and Git.

        EDUCATION

        B.Tech in Artificial Intelligence and Machine Learning.

        PROJECTS

        E-Commerce Sales Dashboard:
        Developed sales dashboards using Excel, SQL,
        Power BI and Tableau. Applied data analysis and
        data visualization techniques to review business KPIs.

        Data Analysis Project:
        Used Python and SQL for data cleaning and exploratory
        data analysis.
        """,
    )

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def main():
    roles = ["Junior Data Analyst"]
    resume = create_test_resume()

    print("\nLOCAL OLLAMA END-TO-END TEST (SYNTHETIC RESUME)")
    print("=" * 55)

    with TestClient(app) as client:
        engine = app.state.matching_engine

        assert engine is not None

        ingestion = process_resume_pdf(
            file_bytes=resume,
            target_roles=roles,
        )

        eligibility = filter_jobs_for_candidate(
            jobs=engine.eligible_jobs,
            target_roles=roles,
            max_required_experience_years=1,
            preferred_country="India",
            candidate_profile=ingestion.profile,
        )

        assert not eligibility.eligible_jobs.empty

        selected = eligibility.eligible_jobs.iloc[0]
        job_id = str(selected["id"])

        print("\nSELECTED ELIGIBLE JOB")
        print("ID:", job_id)
        print("Title:", selected["title"])
        print("Company:", selected["company"])

        response = client.post(
            "/analyze-job",
            files=[
                (
                    "file",
                    (
                        "integration_resume.pdf",
                        resume,
                        "application/pdf",
                    ),
                ),
                ("job_id", (None, job_id)),
                ("target_roles", (None, roles[0])),
            ],
        )

        assert response.status_code == 200, response.text

        result = response.json()
        assert result["required_skill_coverage"] == 1.0, (
            "Synthetic resume should explicitly cover all extracted "
            "required skills."
        )
        assert result["provider"] == "ollama", (
            f"Expected Ollama, received {result['provider']}"
        )
        assert result["used_fallback"] is False

        for skill in (
            result["skill_gaps"] + result["preferred_skill_gaps"]
        ):
            assert skill.casefold() in result["recommendation"].casefold(), (
                f"Recommendation omitted skill gap: {skill}"
            )

        assert result["job_id"] == job_id
        assert result["match_summary"]
        assert result["recommendation"]

        for skill in (
            result["skill_gaps"] + result["preferred_skill_gaps"]
        ):
            assert skill.casefold() in result["recommendation"].casefold(), (
                f"Recommendation omitted canonical skill gap: {skill}"
            )

        assert set(result["matched_skills"]).issubset(
            set(ingestion.profile.skills)
        )

        assert not set(result["matched_skills"]).intersection(
            result["skill_gaps"]
        )

        print("\nANALYSIS RESULT")
        print("Provider:", result["provider"])
        print("Model:", result["model"])
        print("Fallback used:", result["used_fallback"])

        print("\nMATCH SUMMARY")
        print(result["match_summary"])

        print("\nMATCHED REQUIRED SKILLS")
        print(result["matched_skills"])

        print("\nMISSING REQUIRED SKILLS")
        print(result["skill_gaps"])

        print("\nPREFERRED SKILL GAPS")
        print(result["preferred_skill_gaps"])

        print("\nRECOMMENDATION")
        print(result["recommendation"])

        print("\nLOCAL OLLAMA END-TO-END TEST PASSED")
        print("=" * 55)


if __name__ == "__main__":
    main()
