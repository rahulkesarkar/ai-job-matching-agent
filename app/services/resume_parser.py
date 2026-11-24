from dataclasses import dataclass
from io import BytesIO

import pymupdf


@dataclass
class ResumeExtraction:
    text: str
    page_count: int
    extracted_characters: int


class ResumeParsingError(ValueError):
    """Raised when a resume PDF cannot be parsed."""


def extract_resume_text(
    file_bytes: bytes,
    max_file_size_mb: int = 10,
    max_pages: int = 20,
) -> ResumeExtraction:
    """
    Extract text from an uploaded PDF resume.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        max_file_size_mb: Maximum permitted PDF size.
        max_pages: Maximum permitted page count.

    Returns:
        ResumeExtraction containing extracted text and metadata.

    Raises:
        ResumeParsingError: If the PDF is invalid, oversized,
        too long, encrypted, or contains no extractable text.
    """
    if not file_bytes:
        raise ResumeParsingError("The uploaded file is empty.")

    max_file_size = max_file_size_mb * 1024 * 1024

    if len(file_bytes) > max_file_size:
        raise ResumeParsingError(
            f"File exceeds the {max_file_size_mb} MB size limit."
        )

    if not file_bytes.startswith(b"%PDF-"):
        raise ResumeParsingError("The uploaded file is not a valid PDF.")

    try:
        with pymupdf.open(
            stream=BytesIO(file_bytes),
            filetype="pdf",
        ) as document:

            if document.needs_pass:
                raise ResumeParsingError(
                    "Password-protected PDFs are not supported."
                )

            page_count = len(document)

            if page_count == 0:
                raise ResumeParsingError("The PDF contains no pages.")

            if page_count > max_pages:
                raise ResumeParsingError(
                    f"Resume exceeds the {max_pages}-page limit."
                )

            page_texts = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()

                if page_text:
                    page_texts.append(
                        f"--- Page {page_number} ---\n{page_text}"
                    )

            extracted_text = "\n\n".join(page_texts).strip()

    except ResumeParsingError:
        raise
    except Exception as exc:
        raise ResumeParsingError(
            "Unable to read this PDF. Please upload a valid, "
            "unprotected PDF file."
        ) from exc

    if not extracted_text:
        raise ResumeParsingError(
            "No selectable text was found. This may be a scanned "
            "resume; OCR support is required."
        )

    return ResumeExtraction(
        text=extracted_text,
        page_count=page_count,
        extracted_characters=len(extracted_text),
    )
