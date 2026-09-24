from rag.document_manager import (
    create_collection_name,
    create_document_id,
    get_document_name,
)


def test_create_document_id(tmp_path):

    pdf_path = tmp_path / "machine_learning.pdf"

    pdf_path.write_bytes(
        b"test pdf content"
    )

    document_id = create_document_id(
        str(pdf_path)
    )

    assert document_id.startswith("doc_")
    assert len(document_id) == 16


def test_same_file_creates_same_document_id(tmp_path):

    pdf_path = tmp_path / "test.pdf"

    pdf_path.write_bytes(
        b"same content"
    )

    first_id = create_document_id(
        str(pdf_path)
    )

    second_id = create_document_id(
        str(pdf_path)
    )

    assert first_id == second_id


def test_different_content_creates_different_id(tmp_path):

    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"

    first_pdf.write_bytes(
        b"first document"
    )

    second_pdf.write_bytes(
        b"second document"
    )

    first_id = create_document_id(
        str(first_pdf)
    )

    second_id = create_document_id(
        str(second_pdf)
    )

    assert first_id != second_id


def test_create_collection_name():

    collection_name = create_collection_name(
        "doc_123456789abc"
    )

    assert collection_name == "pdf_doc_123456789abc"


def test_get_document_name():

    name = get_document_name(
        "data/uploads/machine_learning.pdf"
    )

    assert name == "machine_learning.pdf"
