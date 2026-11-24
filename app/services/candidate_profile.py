import re
from dataclasses import dataclass, field
from typing import Sequence


SKILL_ALIASES = {
    "Python": ["python"],
    "SQL": ["sql"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "C++": ["c++", "cpp"],
    "C#": ["c#"],
    "HTML": ["html"],
    "CSS": ["css"],
    "React": ["react"],
    "Node.js": ["node.js", "nodejs"],
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "Django": ["django"],
    "REST APIs": ["rest api", "rest apis"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "Keras": ["keras"],
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "Data Analysis": ["data analysis"],
    "Data Cleaning": ["data cleaning"],
    "Exploratory Data Analysis": [
        "exploratory data analysis",
        "eda",
    ],
    "Feature Engineering": ["feature engineering"],
    "Statistics": ["statistics", "statistical analysis"],
    "Data Visualization": ["data visualization"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Docker": ["docker"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "Google Cloud": ["google cloud", "gcp"],
    "Linux": ["linux"],
    "Apache Spark": ["apache spark", "pyspark"],
    "Hadoop": ["hadoop"],
    "Airflow": ["airflow", "apache airflow"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb"],
    "FAISS": ["faiss"],
    "Chroma": ["chroma", "chromadb"],
    "Vector Databases": ["vector database", "vector databases"],
    "Large Language Models": [
        "large language model",
        "large language models",
        "llm",
        "llms",
    ],
    "LangChain": ["langchain"],
    "RAG": ["retrieval-augmented generation", "rag"],
    "Prompt Engineering": ["prompt engineering"],
}


SECTION_ALIASES = {
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career objective",
        "objective",
    ],
    "skills": [
        "skills",
        "technical skills",
        "key skills",
        "core skills",
        "technical competencies",
        "tools and technologies",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "internship experience",
    ],
    "projects": [
        "projects",
        "academic projects",
        "personal projects",
        "relevant projects",
        "project experience",
    ],
    "education": [
        "education",
        "educational background",
        "academic background",
        "academic qualifications",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses and certifications",
    ],
    "target_roles": [
        "target roles",
        "target job titles",
        "desired roles",
        "career interests",
    ],
    "preferences": [
        "career preferences",
        "preferences",
        "job preferences",
        "location preferences",
        "preferred locations",
        "employment preferences",
    ],
}


EDUCATION_PATTERN = re.compile(
    r"\b("
    r"B\.?\s?Tech|BTech|Bachelor(?:'s)?|"
    r"B\.?\s?E\.?|B\.?\s?Sc\.?|"
    r"M\.?\s?Tech|MTech|Master(?:'s)?|"
    r"M\.?\s?E\.?|M\.?\s?Sc\.?|"
    r"Ph\.?\s?D\.?|Doctorate|Diploma"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass
class CandidateProfile:
    skills: list[str] = field(default_factory=list)
    education_evidence: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    target_roles: list[str] = field(default_factory=list)
    matching_text: str = ""


class CandidateProfileError(ValueError):
    """Raised when candidate profile extraction cannot proceed."""


def _normalize_heading(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _identify_section(line: str):
    """Identify a resume heading, including headings with inline content."""
    cleaned = line.strip()

    heading_text = cleaned
    inline_content = ""

    if ":" in cleaned:
        possible_heading, possible_content = cleaned.split(":", 1)
        if len(possible_heading.strip()) <= 55:
            heading_text = possible_heading
            inline_content = possible_content.strip()

    normalized = _normalize_heading(heading_text)

    for section_name, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section_name, inline_content

    return None



def clean_resume_text(resume_text: str) -> str:
    """
    Remove common cover-letter template instructions and placeholders
    before extracting candidate information.
    """
    if not isinstance(resume_text, str):
        raise CandidateProfileError("Resume text must be a string.")

    cleaned_lines = []

    for line in resume_text.splitlines():
        cleaned = line.strip()

        if not cleaned:
            continue

        # A cover-letter template appended to a resume must not become
        # candidate evidence. Preserve the actual resume text preceding it.
        template_match = re.search(
            r"\b(?:what draws me to|and i write)\b",
            cleaned,
            flags=re.IGNORECASE,
        )

        if template_match:
            cleaned = cleaned[:template_match.start()].strip()

        # Remove standalone template guidance and placeholder instructions.
        if re.search(
            r"\b(one specific, verifiable fact|"
            r"product launch|system migration|market expansion)\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            continue

        # Remove common bracketed placeholders without discarding
        # valid text elsewhere on the same line.
        cleaned = re.sub(
            r"\[(?:company|system|one specific[^\]]*)\]",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")

        if cleaned:
            cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines)


def extract_resume_sections(resume_text: str) -> dict[str, str]:
    """Extract common resume sections using recognizable headings."""
    sections = {}
    current_section = None

    for line in resume_text.splitlines():
        cleaned = line.strip()

        if not cleaned:
            continue

        identified = _identify_section(cleaned)

        if identified:
            current_section, inline_content = identified
            sections.setdefault(current_section, [])

            if inline_content:
                sections[current_section].append(inline_content)

            continue

        if current_section:
            sections[current_section].append(cleaned)

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
        if lines
    }


def extract_skills(resume_text: str) -> list[str]:
    """Find skills explicitly mentioned in the supplied resume text."""
    found_skills = []

    for canonical_name, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(alias)
                + r"(?![A-Za-z0-9])"
            )

            if re.search(pattern, resume_text, flags=re.IGNORECASE):
                found_skills.append(canonical_name)
                break

    return found_skills


def extract_education_evidence(resume_text: str) -> list[str]:
    """
    Extract formal qualifications only from the recognized Education section.

    Preserve complete qualification lines, including institution, dates,
    specialization, grades and original punctuation.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        return []

    sections = extract_resume_sections(resume_text)
    education_text = sections.get("education", "")

    if not education_text:
        return []

    evidence = []

    for line in education_text.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"^[•*\-]+\s*", "", cleaned)

        if not cleaned or not EDUCATION_PATTERN.search(cleaned):
            continue

        # Remove a separate certification clause, not degree specializations.
        cleaned = re.split(
            r"\s+(?:and\s+|or\s+)?(?:AWS\s+)?Certified\b"
            r"|\s+(?:and\s+|or\s+)?Certification\b"
            r"|\s+(?:and\s+|or\s+)?Certificate\b",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned and cleaned not in evidence:
            evidence.append(cleaned)

    return evidence


def build_candidate_profile(
    resume_text: str,
    target_roles: Sequence[str] | None = None,
) -> CandidateProfile:
    """
    Convert extracted resume text into a structured candidate profile.

    Target roles are supplied separately because they are user preferences,
    not qualifications that should be inferred from a resume.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise CandidateProfileError("Resume text cannot be empty.")

    resume_text = clean_resume_text(resume_text)

    if not resume_text.strip():
        raise CandidateProfileError(
            "No usable resume text remained after cleaning."
        )

    skills = extract_skills(resume_text)
    sections = extract_resume_sections(resume_text)
    education = extract_education_evidence(resume_text)

    roles = [
        role.strip()
        for role in (target_roles or [])
        if isinstance(role, str) and role.strip()
    ]

    # Use relevant professional content for matching.
    # Do not mix location or employment preferences into text similarity.
    matching_parts = []

    for section_name in [
        "summary",
        "skills",
        "experience",
        "projects",
        "certifications",
    ]:
        content = sections.get(section_name)

        if content:
            matching_parts.append(content)

    if education:
        matching_parts.extend(education)

    if roles:
        matching_parts.append("Target roles: " + ", ".join(roles))

    # Fallback for resumes without recognizable section headings.
    if not matching_parts:
        matching_parts.append(resume_text.strip())

    matching_text = "\n\n".join(matching_parts)

    return CandidateProfile(
        skills=skills,
        education_evidence=education,
        sections=sections,
        target_roles=roles,
        matching_text=matching_text,
    )
