from pathlib import Path

import pytest

from rag.ingestion import load_pdf


def test_load_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent.pdf")


def test_load_pdf_requires_pdf_extension(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError):
        load_pdf(str(file_path))
