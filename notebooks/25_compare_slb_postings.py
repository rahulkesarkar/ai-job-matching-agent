import html
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "jobs_cleaned.parquet"

SLB_IDS = [
    "linkedin_4404127001",
    "linkedin_4404112449",
    "linkedin_4404120063",
    "linkedin_4404110944",
]


def clean_text(text):
    text = html.unescape(str(text))
    text = text.replace("<![CDATA[", "")
    text = text.replace("]]>", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def main():

    print("\nSLB POSTING COMPARISON")
    print("=" * 65)

    jobs = pd.read_parquet(DATA_PATH)

    selected = jobs[jobs["id"].isin(SLB_IDS)].copy()

    if len(selected) != 4:
        raise ValueError(
            f"Expected 4 SLB postings, found {len(selected)}"
        )

    selected["clean_description"] = (
        selected["description"].map(clean_text)
    )

    print("\n1. POSTING METADATA")
    print("-" * 65)

    for _, row in selected.iterrows():
        print("\nJob ID:", row["id"])
        print("Title:", row["title"])
        print("Company:", row["company"])
        print("Location:", row["location"])
        print("Seniority:", row["seniority_level"])
        print("Description characters:", len(row["description"]))
        print("URL:", row["url"])

    # Compare the two representatives from our top 10.
    first = selected.loc[
        selected["id"] == "linkedin_4404127001"
    ].iloc[0]["clean_description"].split()

    second = selected.loc[
        selected["id"] == "linkedin_4404120063"
    ].iloc[0]["clean_description"].split()

    matcher = SequenceMatcher(
        None, first, second, autojunk=False
    )

    print("\n2. COMPARISON OF THE TWO RECOMMENDATIONS")
    print("-" * 65)

    print("Description A word count:", len(first))
    print("Description B word count:", len(second))
    print("Text similarity ratio:", round(matcher.ratio(), 4))

    print("\n3. DIFFERENCES BETWEEN DESCRIPTIONS")
    print("-" * 65)

    difference_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            continue

        difference_count += 1

        before = " ".join(first[i1:i2])[:350]
        after = " ".join(second[j1:j2])[:350]

        print(f"\nDifference {difference_count}: {tag}")
        print("First posting:", before or "[No text]")
        print("Second posting:", after or "[No text]")

        if difference_count >= 10:
            print("\nFurther differences omitted.")
            break

    if difference_count == 0:
        print("Descriptions are identical after normalization.")

    print("\n4. CONCLUSION DATA")
    print("-" * 65)

    print("Exact normalized descriptions:", first == second)
    print(
        "Same company:",
        selected["company"].nunique() == 1
    )
    print(
        "Same title:",
        selected["title"].nunique() == 1
    )
    print(
        "Same original location:",
        selected["location"].nunique() == 1
    )

    print("\nCOMPARISON COMPLETED")


if __name__ == "__main__":
    main()
