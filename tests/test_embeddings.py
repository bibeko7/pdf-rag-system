from rag.embeddings import (
    EMBEDDING_MODEL_NAME,
    create_embeddings,
)


def test_embedding_model_name():
    assert EMBEDDING_MODEL_NAME == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_create_embeddings():
    embeddings = create_embeddings()

    vector = embeddings.embed_query(
        "What is machine learning?"
    )

    assert isinstance(vector, list)
    assert len(vector) > 0
