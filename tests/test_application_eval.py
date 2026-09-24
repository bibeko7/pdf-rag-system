from pathlib import Path

import pytest

from rag.document_manager import create_document_id
from rag.pipeline import RAGPipeline


def test_application_rejects_question_before_indexing():
    pipeline = RAGPipeline()

    with pytest.raises(RuntimeError, match="Call ingest"):
        pipeline.ask("What is machine learning?")


def test_application_rejects_empty_question():
    pipeline = RAGPipeline()

    pipeline.retriever = object()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask("   ")


def test_application_rejects_missing_pdf():
    pipeline = RAGPipeline()

    with pytest.raises(FileNotFoundError):
        pipeline.ingest("does_not_exist.pdf")


def test_application_rejects_non_pdf(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("not a pdf", encoding="utf-8")

    pipeline = RAGPipeline()

    with pytest.raises(ValueError, match="Only PDF files"):
        pipeline.ingest(str(file_path))


def test_document_identity_is_content_based(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"

    content = b"same document content"

    first.write_bytes(content)
    second.write_bytes(content)

    assert create_document_id(str(first)) == create_document_id(str(second))


def test_application_ingestion_creates_retriever():
    pdf_path = Path("data/evaluation/Machine_Learning_No_Table.pdf")

    pipeline = RAGPipeline()

    chunks = pipeline.ingest(str(pdf_path))

    assert chunks > 0
    assert pipeline.retriever is not None
    assert pipeline.vectorstore is not None
    assert pipeline.document_id is not None
    assert pipeline.document_name == pdf_path.name


def test_application_retrieval_returns_documents():
    pdf_path = Path("data/evaluation/Machine_Learning_No_Table.pdf")

    pipeline = RAGPipeline()
    pipeline.ingest(str(pdf_path))

    documents = pipeline.retriever.invoke(
        "What is Machine Learning?"
    )

    assert len(documents) > 0

    for document in documents:
        assert document.page_content.strip()
        assert "source" in document.metadata
        assert "page" in document.metadata


def test_application_uses_mmr_retrieval():
    pdf_path = Path("data/evaluation/Machine_Learning_No_Table.pdf")

    pipeline = RAGPipeline()
    pipeline.ingest(str(pdf_path))

    assert pipeline.retriever.search_type == "mmr"
