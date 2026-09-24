from langchain_core.vectorstores import VectorStoreRetriever


DEFAULT_K = 4
DEFAULT_FETCH_K = 12


def create_retriever(
    vectorstore,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_FETCH_K,
) -> VectorStoreRetriever:
    """
    Create an MMR retriever from a vector store.

    Args:
        vectorstore: Chroma vector store.
        k: Number of final documents returned.
        fetch_k: Number of candidate documents considered by MMR.

    Returns:
        Configured vector store retriever.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if fetch_k < k:
        raise ValueError("fetch_k must be greater than or equal to k.")

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
        },
    )
