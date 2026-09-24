from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings():
    """
    Create the embedding model used by the RAG system.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )
