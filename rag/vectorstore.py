from langchain_community.vectorstores import Chroma

CHROMA_DIRECTORY = "./chroma_db"


def create_vectorstore(
    chunks,
    embeddings,
    collection_name: str,
    persist_directory: str = CHROMA_DIRECTORY,
):
    if not chunks:
        raise ValueError("Cannot create a vector store from empty chunks.")

    if not collection_name:
        raise ValueError("collection_name cannot be empty.")

    if not persist_directory:
        raise ValueError("persist_directory cannot be empty.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return vectorstore
