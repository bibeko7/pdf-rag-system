import uuid

from langchain_core.documents import Document

from rag.embeddings import create_embeddings
from rag.vectorstore import create_vectorstore
from rag.retriever import create_retriever


def test_create_retriever():

    embeddings = create_embeddings()

    documents = [
        Document(
            page_content="Machine learning learns patterns from data.",
            metadata={"source": "test.pdf", "page": 0}
        ),
        Document(
            page_content="Supervised learning uses labeled data.",
            metadata={"source": "test.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={"source": "test.pdf", "page": 2}
        ),
    ]

    collection_name = f"retriever_test_{uuid.uuid4().hex[:12]}"

    vectorstore = create_vectorstore(
        documents,
        embeddings,
        collection_name
    )

    retriever = create_retriever(
        vectorstore,
        k=2,
        fetch_k=3
    )

    results = retriever.invoke(
        "What is machine learning?"
    )

    assert len(results) == 2
    assert all(
        hasattr(document, "page_content")
        for document in results
    )


def test_retriever_rejects_invalid_k():

    embeddings = create_embeddings()

    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf", "page": 0}
        )
    ]

    collection_name = f"retriever_test_{uuid.uuid4().hex[:12]}"

    vectorstore = create_vectorstore(
        documents,
        embeddings,
        collection_name
    )

    try:
        create_retriever(
            vectorstore,
            k=0
        )
        assert False
    except ValueError:
        assert True


def test_retriever_rejects_invalid_fetch_k():

    embeddings = create_embeddings()

    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf", "page": 0}
        )
    ]

    collection_name = f"retriever_test_{uuid.uuid4().hex[:12]}"

    vectorstore = create_vectorstore(
        documents,
        embeddings,
        collection_name
    )

    try:
        create_retriever(
            vectorstore,
            k=4,
            fetch_k=2
        )
        assert False
    except ValueError:
        assert True
