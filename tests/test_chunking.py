from langchain_core.documents import Document

from rag.chunking import split_documents


def test_split_documents_creates_chunks():

    documents = [
        Document(
            page_content=(
                "Machine learning is a field of artificial intelligence. "
                "It allows computers to learn patterns from data. "
                "Models can be trained using different algorithms."
            ),
            metadata={
                "source": "test.pdf",
                "page": 0
            }
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) > 0
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_split_documents_preserves_metadata():

    documents = [
        Document(
            page_content="This is a test document about machine learning.",
            metadata={
                "source": "test.pdf",
                "page": 3
            }
        )
    ]

    chunks = split_documents(documents)

    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["page"] == 3
