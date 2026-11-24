import html
import re
from dataclasses import dataclass

import pandas as pd


ROLE_PATTERNS = {
    "data_analyst": [
        r"\bdata analyst\b",
        r"\bbusiness intelligence analyst\b",
        r"\bbi analyst\b",
        r"\breporting analyst\b",
        r"\bmis analyst\b",
    ],
    "data_scientist": [
        r"\bdata scientist\b",
        r"\bdata science\b",
    ],
    "ml_engineer": [
        r"\bmachine learning engineer\b",
        r"\bml engineer\b",
        r"\bai/?ml engineer\b",
        r"\bartificial intelligence engineer\b",
        r"\bai engineer\b",
        r"\bai and data engineer\b",
        r"\bai\b.{0,30}\bengineer\b",
    ],
}


SENIOR_TITLE_PATTERN = re.compile(
    r"\b(senior|sr\.?|lead|manager|director|principal|"
    r"head|chief|executive|vice president|vp)\b",
    re.IGNORECASE,
)

EARLY_CAREER_PATTERN = re.compile(
    r"\b(freshers?|fresh graduates?|recent graduates?|"
    r"entry[- ]level|internship|intern|early[- ]career|"
    r"graduate role|new graduates?)\b|"
    r"\b0\s*(?:-|–|—|to)\s*1\s*years?\b",
    re.IGNORECASE,
)

RANGE_EXPERIENCE_PATTERN = re.compile(
    r"\b(?P<minimum>\d{1,2})\s*"
    r"(?:-|–|—|to)\s*\d{1,2}\+?\s*years?\b",
    re.IGNORECASE,
)

SINGLE_EXPERIENCE_PATTERN = re.compile(
    r"\b(?P<minimum>\d{1,2})\+?\s*years?\b",
    re.IGNORECASE,
)

EXPERIENCE_REQUIREMENT_CUE = re.compile(
    r"\b(required|requires?|minimum|min\.?|at least|"
    r"must have|must possess|typically|you have|you bring|"
    r"candidate with|professional with|qualified candidate|"
    r"ideal candidate|we are seeking|looking for|professional|"
    r"hands[- ]on)\b|"
    r"\bexperience\s*[:\-]",
    re.IGNORECASE,
)

COMPANY_BACKGROUND_CUE = re.compile(
    r"\\b(combined experience|collective experience|company history|"
    r"company heritage|founded|over \\d+ years|more than \\d+ years|"
    r"with over \\d+ years|\\d+ years of combined experience)\\b",
    re.IGNORECASE,
)

ADVANCED_DEGREE_PATTERNS = {
    "PhD": re.compile(
        r"\b(ph\.?\s?d\.?|doctorate|doctoral degree)\b",
        re.IGNORECASE,
    ),
    "Master's degree": re.compile(
        r"\b(master(?:'s)?(?: of science| of engineering)?|"
        r"m\.?\s?sc\.?|m\.?\s?tech\.?|m\.?\s?e\.?)\b",
        re.IGNORECASE,
    ),
}

DEGREE_PREFERENCE_PATTERN = re.compile(
    r"\b(preferred|preferable|desirable|ideally|"
    r"nice to have|a plus|an advantage|bonus)\b",
    re.IGNORECASE,
)

DEGREE_REQUIREMENT_PATTERN = re.compile(
    r"\b(mandatory|required|must have|must possess|"
    r"requires?|minimum qualification|minimum requirement)\b",
    re.IGNORECASE,
)


@dataclass
class EligibilityResult:
    eligible_jobs: pd.DataFrame
    excluded_jobs: pd.DataFrame
    total_jobs: int
    eligible_count: int
    excluded_count: int


class JobEligibilityError(ValueError):
    """Raised when eligibility filtering cannot be performed."""


def normalize_text(value) -> str:
    """Normalize text for predictable matching."""
    if pd.isna(value):
        return ""

    value = html.unescape(str(value))
    value = re.sub(r"\s*&\s*", " and ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


def identify_target_categories(target_roles: list[str]) -> list[str]:
    """Map selected user roles to supported recommendation categories."""
    categories = []

    for role in target_roles:
        role_text = normalize_text(role)

        if "data analyst" in role_text:
            category = "data_analyst"
        elif "data scientist" in role_text or "data science" in role_text:
            category = "data_scientist"
        elif (
            "machine learning" in role_text
            or re.search(r"\bml engineer\b", role_text)
            or "ai engineer" in role_text
        ):
            category = "ml_engineer"
        else:
            continue

        if category not in categories:
            categories.append(category)

    return categories


def match_target_role(
    title: str,
    categories: list[str],
) -> str | None:
    """Match job title against explicitly selected candidate roles."""
    normalized_title = normalize_text(title)

    for category in categories:
        for pattern in ROLE_PATTERNS[category]:
            if re.search(pattern, normalized_title, re.IGNORECASE):
                return category

    return None


def _text_segments(text: str) -> list[str]:
    """Split descriptions into lines and sentence-like segments."""
    segments = []

    for line in text.splitlines():
        for segment in re.split(r"(?<=[.!?;])\s+", line):
            segment = segment.strip()
            if segment:
                segments.append(segment)

    return segments


def extract_minimum_required_experience(text: str) -> float | None:
    """
    Extract minimum experience from actual job requirements.

    Supports:
    - 4+ years of experience
    - 2-4 years of professional experience
    - Experience: 2-4 years
    - Standard I: Two (2) years, when preceded by
      an explicit work-experience grading section.

    Company-history statements are ignored.
    """
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    range_pattern = re.compile(
        r"\b(?P<minimum>\d{1,2})\s*"
        r"(?:-|–|—|to)\s*\d{1,2}\+?\s*years?\b",
        re.IGNORECASE,
    )

    word_number_pattern = re.compile(
        r"\b(?P<word>one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s*\(\s*(?P<digit>\d{1,2})\s*\)\s*years?\b",
        re.IGNORECASE,
    )

    single_pattern = re.compile(
        r"\b(?P<minimum>\d{1,2})\+?\s*years?\b",
        re.IGNORECASE,
    )

    word_year_pattern = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\b",
        re.IGNORECASE,
    )

    detected = []
    standard_experience_context = False

    for segment in _text_segments(text):
        normalized = normalize_text(segment)

        if COMPANY_BACKGROUND_CUE.search(normalized):
            continue

        if re.search(
            r"\b(relevant work experience|work experience)\b"
            r".*\b(following number|standard|experience level)\b",
            normalized,
        ):
            standard_experience_context = True

        has_experience_word = bool(
            re.search(r"\bexperience\b", normalized)
        )

        is_standard_requirement = (
            standard_experience_context
            and bool(
                re.search(
                    r"\b(?:standard|senior)\s+(?:i{1,3}|iv|v|[1-5])\b",
                    normalized,
                )
            )
        )

        if not has_experience_word and not is_standard_requirement:
            continue

        matches = list(range_pattern.finditer(normalized))

        if matches:
            detected.extend(
                float(match.group("minimum"))
                for match in matches
            )
            continue

        word_match = word_number_pattern.search(normalized)

        if word_match:
            detected.append(float(word_match.group("digit")))
            continue

        single_matches = list(single_pattern.finditer(normalized))

        if single_matches:
            detected.extend(
                float(match.group("minimum"))
                for match in single_matches
            )
            continue

        word_match = word_year_pattern.search(normalized)

        if word_match:
            detected.append(
                float(number_words[word_match.group(1).lower()])
            )

    valid_values = [
        value for value in detected
        if 0 <= value <= 40
    ]

    return max(valid_values) if valid_values else None


def extract_required_advanced_degree(text: str) -> str | None:
    """
    Detect mandatory Master's or PhD requirements.

    Preferred qualifications alone are not treated as mandatory.
    Research/thesis-based Master's requirements are treated as
    required when they are explicitly described as qualifications.
    """
    for segment in _text_segments(text):
        normalized = normalize_text(segment)

        has_preference = bool(
            DEGREE_PREFERENCE_PATTERN.search(normalized)
        )

        if has_preference:
            continue

        matched_degree = None

        for degree_name, pattern in ADVANCED_DEGREE_PATTERNS.items():
            if pattern.search(normalized):
                matched_degree = degree_name
                break

        if matched_degree is None:
            continue

        explicitly_required = bool(
            DEGREE_REQUIREMENT_PATTERN.search(normalized)
        )

        research_degree = (
            matched_degree == "Master's degree"
            and bool(
                re.search(
                    r"\b(thesis|research projects?|research experience)\b",
                    normalized,
                    re.IGNORECASE,
                )
            )
        )

        if explicitly_required or research_degree:
            return matched_degree

    return None


def is_senior_job(
    title: str,
    seniority: str,
    description: str = "",
) -> bool:
    """
    Reject clearly senior titles and leadership metadata.

    Explicit early-career evidence can override generic Mid-Senior
    metadata, but cannot override a clearly senior job title.
    """
    title_text = normalize_text(title)
    seniority_text = normalize_text(seniority)

    if SENIOR_TITLE_PATTERN.search(title_text):
        return True

    if seniority_text in {"director", "executive"}:
        return True

    if seniority_text == "mid-senior level":
        description_text = normalize_text(description)

        explicit_early_career = bool(
            EARLY_CAREER_PATTERN.search(description_text)
        )

        junior_in_title = bool(
            re.search(r"\\bjunior\\b", title_text)
        )

        return not (explicit_early_career or junior_in_title)

    return False



def supports_academic_project_equivalency(text: str) -> bool:
    """Check whether the employer explicitly accepts academic/project equivalency."""
    normalized = normalize_text(text)

    patterns = [
        r"\bor\s+equivalent\s+academic\s+and\s+project\s+experience\b",
        r"\bequivalent\s+(?:academic|educational)\s+and\s+project\s+experience\b",
        r"\bequivalent\s+combination\s+of\s+(?:education|academic)"
        r".{0,50}\b(?:project|practical)\b",
    ]

    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in patterns
    )


def candidate_has_academic_project_evidence(candidate_profile) -> bool:
    """
    Require explicit bachelor's-level education and substantive project work.

    This does not estimate or invent professional employment duration.
    """
    if candidate_profile is None:
        return False

    education = getattr(candidate_profile, "education_evidence", [])
    sections = getattr(candidate_profile, "sections", {})

    has_bachelor_qualification = any(
        re.search(
            r"\b(b\.?\s?tech|btech|b\.?\s?e\.?|bachelor(?:'s)?)\b",
            normalize_text(item),
        )
        for item in education
    )

    projects = sections.get("projects", "")
    has_substantive_projects = (
        isinstance(projects, str)
        and len(projects.strip()) >= 40
    )

    return has_bachelor_qualification and has_substantive_projects


def filter_jobs_for_candidate(
    jobs: pd.DataFrame,
    target_roles: list[str],
    max_required_experience_years: float = 1,
    preferred_country: str = "India",
    candidate_profile=None,
) -> EligibilityResult:
    """
    Filter jobs by target role, country, seniority,
    experience and mandatory advanced qualifications.
    """
    required_columns = {
        "id",
        "title",
        "company",
        "location",
        "description",
        "seniority_level",
        "normalized_country",
    }

    missing = required_columns - set(jobs.columns)

    if missing:
        raise JobEligibilityError(
            f"Missing job columns: {sorted(missing)}"
        )

    if max_required_experience_years < 0:
        raise JobEligibilityError(
            "Maximum required experience cannot be negative."
        )

    categories = identify_target_categories(target_roles)

    if not categories:
        raise JobEligibilityError(
            "Select at least one supported target role: "
            "Data Analyst, Data Scientist or Machine Learning Engineer."
        )

    eligible_records = []
    excluded_records = []

    for _, row in jobs.iterrows():
        title = row["title"]
        description = str(row["description"])
        seniority = row["seniority_level"]

        role_category = match_target_role(title, categories)

        required_years = extract_minimum_required_experience(
            f"{title}\n{description}"
        )

        advanced_degree = extract_required_advanced_degree(
            f"{title}\n{description}"
        )

        reasons = []

        country = normalize_text(row["normalized_country"])

        if country != normalize_text(preferred_country):
            reasons.append(
                "Outside preferred country or country unclassified"
            )

        if role_category is None:
            reasons.append(
                "Job title does not match selected target roles"
            )

        if is_senior_job(title, seniority, description):
            reasons.append("Senior or leadership position")

        equivalency_applied = (
            required_years is not None
            and required_years > max_required_experience_years
            and supports_academic_project_equivalency(
                f"{title}\n{description}"
            )
            and candidate_has_academic_project_evidence(
                candidate_profile
            )
        )

        if (
            required_years is not None
            and required_years > max_required_experience_years
            and not equivalency_applied
        ):
            reasons.append(
                f"Requires at least {required_years:g} years of experience"
            )

        if advanced_degree is not None:
            reasons.append(
                f"Requires a mandatory {advanced_degree}"
            )

        record = row.to_dict()
        record["matched_target_role"] = role_category
        record["required_experience_min_years"] = required_years
        record["required_advanced_degree"] = advanced_degree
        record["experience_equivalency_applied"] = equivalency_applied

        if reasons:
            record["exclusion_reasons"] = reasons
            excluded_records.append(record)
        else:
            eligible_records.append(record)

    extra_columns = [
        "matched_target_role",
        "required_experience_min_years",
        "required_advanced_degree",
        "experience_equivalency_applied",
    ]

    eligible = pd.DataFrame(
        eligible_records,
        columns=list(jobs.columns) + extra_columns,
    )

    excluded = pd.DataFrame(
        excluded_records,
        columns=list(jobs.columns) + extra_columns + [
            "exclusion_reasons"
        ],
    )

    return EligibilityResult(
        eligible_jobs=eligible.reset_index(drop=True),
        excluded_jobs=excluded.reset_index(drop=True),
        total_jobs=len(jobs),
        eligible_count=len(eligible),
        excluded_count=len(excluded),
    )
