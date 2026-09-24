from pathlib import Path

import pytest

from rag.vectorstore import create_vectorstore


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeDocument:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}


def test_create_vectorstore_with_custom_directory(tmp_path):
    chunks = [
        FakeDocument("Machine learning is a field of AI.")
    ]

    vectorstore = create_vectorstore(
        chunks=chunks,
        embeddings=FakeEmbeddings(),
        collection_name="test_collection",
        persist_directory=str(tmp_path / "chroma_test"),
    )

    assert vectorstore is not None
    assert Path(tmp_path / "chroma_test").exists()


def test_create_vectorstore_rejects_empty_chunks():
    with pytest.raises(ValueError):
        create_vectorstore(
            chunks=[],
            embeddings=FakeEmbeddings(),
            collection_name="test_collection",
        )


def test_create_vectorstore_rejects_empty_collection_name():
    chunks = [
        FakeDocument("Test document")
    ]

    with pytest.raises(ValueError):
        create_vectorstore(
            chunks=chunks,
            embeddings=FakeEmbeddings(),
            collection_name="",
        )


def test_create_vectorstore_rejects_empty_persist_directory():
    chunks = [
        FakeDocument("Test document")
    ]

    with pytest.raises(ValueError):
        create_vectorstore(
            chunks=chunks,
            embeddings=FakeEmbeddings(),
            collection_name="test_collection",
            persist_directory="",
        )
