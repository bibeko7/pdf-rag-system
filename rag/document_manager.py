import hashlib
import re
from pathlib import Path


def create_document_id(pdf_path: str) -> str:
    """
    Create a stable document ID from the PDF file path and filename.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    file_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()[:12]

    return f"doc_{file_hash}"


def create_collection_name(document_id: str) -> str:
    """
    Convert a document ID into a valid Chroma collection name.
    """

    if not document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    safe_id = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        document_id
    )

    return f"pdf_{safe_id}"


def get_document_name(pdf_path: str) -> str:
    """
    Return the original PDF filename.
    """

    return Path(pdf_path).name
